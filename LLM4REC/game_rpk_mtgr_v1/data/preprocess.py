import gc
import json
import logging
import os
import stat
from multiprocessing import Pool
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from absl import app, flags
import pyarrow.parquet as pq

# ========= 配置参数 =========

logging.basicConfig(level=logging.INFO)
FLAGS = flags.FLAGS 

flags.DEFINE_string("INPUT_DIR", None, "Path to the dataset folder containing the original.", required=True)
flags.DEFINE_string("OUTPUT_DIR", None, "Path to the dataset output folder for processed data.", required=True)
flags.DEFINE_string("config_file", None, "Path to the config file containing the parameters.", required=True)



def read_json(json_path: str) -> Dict[str, Any]:
    """
    读取json格式文件
    :param json_path: 文件路径
    :return:
    """
    read_flags = os.O_RDONLY
    modes = stat.S_IRUSR

    with os.fdopen(os.open(json_path, read_flags, modes), 'r', encoding='UTF-8') as f:
        result = json.loads(f.read())
    return result


def remove_zero_label_by_index(df, label_col='rpk_active_flag', remove_ratio=0.8,
    pos_neg_rat=False):
    """
    按 pt_d 和 did 分组，按指定比例删除指定标签列中值为 0 的样本（通过索引直接删除）
    
    参数:
        df: 原始 DataFrame，需包含 pt_d、did 和指定的标签列
        remove_ratio: 要删除的标签值为 0 的样本比例 (0-1)，例如 0.5 表示删除50%
        label_col: 标签列的列名，默认为 'rpk_active_flag'
    
    返回:
        处理后的 DataFrame
    """
    # 检查必要的列是否存在
    required_cols = ['pt_d', 'did', label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 缺少必要的列: {missing_cols}")
    
    # 1. 筛选出所有指定标签列值为 0 的样本
    mask_label_0 = df[label_col] == 0
    df_label_0 = df[mask_label_0].copy()
    
    if df_label_0.empty:
        return df.copy()
    
    # 2. 按 pt_d 和 did 分组，获取需要删除的索引
    remove_indices = []

    def get_remove_indices(group):
        # 计算需要删除的样本数量
        n_remove = int(len(group) * remove_ratio)
        if n_remove == 0:
            return
        # 随机选取需要删除的索引并添加到列表
        remove_indices.extend(group.sample(n=n_remove, random_state=42).index.tolist())
    
    # 对每个分组应用函数，收集需要删除的索引
    df_label_0.groupby(['pt_d', 'did'], group_keys=False).apply(get_remove_indices)
    
    # 3. 从原数据中删除指定索引的样本
    df = df.drop(remove_indices)
    # 重置索引（可选，保证索引连续）
    df = df.reset_index(drop=True)
    
    return df


def remove_zero_label_global(
    df,
    label_col='rpk_active_flag',
    remove_ratio=0.5,
    pos_neg_rat=False
):
    """
    对整个数据集整体抽样，按指定比例删除指定标签列中值为 0 的样本（通过索引直接删除）
    
    参数:
        df: 原始 DataFrame，需包含指定的标签列
        remove_ratio: 要删除的标签值为 0 的样本比例 (0-1)，例如 0.5 表示删除50%
        label_col: 标签列的列名，默认为 'rpk_active_flag'
    
    返回:
        处理后的 DataFrame
    """
    # 检查必要的列是否存在
    if label_col not in df.columns:
        raise ValueError(f"DataFrame 缺少指定的标签列: {label_col}")
    
    # 1. 筛选出所有指定标签列值为 0 的样本
    mask_label_0 = df[label_col] == 0
    df_label_0 = df[mask_label_0]
    
    if df_label_0.empty:
        logging.info("提示：没有 %s=0 的样本需要删除", label_col)
        return df.copy()
    
    # 2. 计算需要删除的样本数量并随机选取索引
    if pos_neg_rat:
        n_pos = len(df[df[label_col] == 1])
        n_remove = len(df_label_0) - int(n_pos / remove_ratio)
    else:
        n_remove = int(len(df_label_0) * remove_ratio)
    if n_remove == 0:
        logging.info("提示：按 %s 比例计算，无需删除任何 %s=0 的样本", remove_ratio, label_col)
        return df.copy()
    
    # 随机选取需要删除的索引
    remove_indices = df_label_0.sample(n=n_remove, random_state=42).index
    
    # 3. 从原数据中删除指定索引的样本
    df = df.drop(remove_indices)
    # 重置索引（可选，保证索引连续）
    df = df.reset_index(drop=True)
    
    return df


def get_hist_index(hist_date: list, cand_date: int) -> list:
    """
    优化版1：numpy向量化操作（效率最高）
    """
    # 转换为numpy数组（一次性完成类型转换，比循环内转换快）
    hist_arr = np.array(hist_date, dtype=int)
    # 向量化筛选条件：< cand_date 且 != 0
    mask = (hist_arr < cand_date) & (hist_arr != 0)
    # 获取满足条件的索引并转为列表
    valid_indices = np.where(mask)[0].tolist()
    return valid_indices


def read_parquet_to_dataframe_safe(file_path, hist_cols, cand_cols, ctype_require=None, num_examples=None):
    df = pq.read_table(file_path).to_pandas()
    for col in cand_cols:
        df[col] = df[col].map(lambda x: int(x[0]))
        
    return df


def to_2d_list(series):
    """将Series中的每个元素包装成列表，返回二维列表"""
    # 如果元素本身是list/np.array，直接放入外层列表；如果是单个值，先包装成列表
    return [list(item) if isinstance(item, (list, np.ndarray)) else item for item in series]


def optimized_agg_for_multi_groupkeys(df, group_keys, first_cols, list_cols):
    # 步骤1：统计每个多列分组的行数（核心适配：基于多列groupkey）
    # 方法：添加临时列标记分组，或用groupby.size（多列需指定group_keys）
    group_size = df.groupby(group_keys).size().reset_index(name='count')
    # 合并分组大小到原数据，方便筛选（比多次isin更快）
    df_with_count = df.merge(group_size, on=group_keys, how='left')
    
    # 步骤2：处理单条数据分组（count=1）
    df_single_result = df_with_count[df_with_count['count'] == 1].copy()
    # 格式对齐：单条数据的list_col转为二维列表
    for col in list_cols:
        df_single_result[col] = df_single_result[col].apply(lambda x: [x])
    # 删除临时count列
    df_single_result = df_single_result.drop(columns=['count'])
    
    # 步骤3：处理多条数据分组（count>1）
    df_multi_data = df_with_count[df_with_count['count'] > 1].copy()
    df_multi_data = df_multi_data.drop(columns=['count'])  # 删除临时列
    # 仅对多条数据分组执行groupby+agg
    agg_rules = {}
    for col in first_cols:
        agg_rules[col] = 'first'
    for col in list_cols:
        agg_rules[col] = list
    df_multi_result = df_multi_data.groupby(group_keys, as_index=False).agg(agg_rules)

    # 步骤4：合并结果（保持列顺序一致）
    df_merged = pd.concat([df_single_result, df_multi_result], ignore_index=True)
    # 按多列groupkey排序（可选，与原始结果对齐）
    df_merged = df_merged.sort_values(group_keys).reset_index(drop=True)
    
    return df_merged


def process_file(ipt_config: dict):
    """处理单个文件的函数"""
    file_name = ipt_config.get("file_name")
    input_dir = ipt_config.get("input_dir")
    output_dir = ipt_config.get("output_dir")
    file_end = ipt_config.get("file_end")
    user_cols = ipt_config.get("user_cols")
    cross_cols = ipt_config.get("cross_cols")
    history_cols = ipt_config.get("history_cols")
    candidate_cols = ipt_config.get("candidate_cols")
    group_keys = ipt_config.get("group_keys")
    opt_file_end = ipt_config.get("opt_file_end")
    batch_size = ipt_config.get("batch_size", 0)
    col_names = ipt_config.get("col_names")
    oper_time_name = col_names.get("oper_time", "oper_time")
    oper_date_name = col_names.get("oper_date", "oper_date")
    hist_date_name = col_names.get("history_date", "history_date")
    opt_file_path = os.path.join(output_dir, file_name.replace(file_end, opt_file_end))
    batch_first_name = f"{file_name.replace(file_end,'')}_1{opt_file_end}"
    if os.path.isfile(opt_file_path) or os.path.isfile(os.path.join(output_dir, batch_first_name)):
        logging.info("File already exists, skipping: %s", opt_file_path)
        return

    logging.info("reading_file: %s", file_name)
    file_path = os.path.join(input_dir, file_name)
    df = read_parquet_to_dataframe_safe(file_path, history_cols, candidate_cols)
    
    if df.empty:
        logging.info("Empty DataFrame from %s, skipping...", file_name)
        return
    logging.info("Processing file: %s", file_name)
    logging.info("before_shape %s", df.shape)
    df = remove_zero_label_global(
        df,
        ipt_config.get("label_col"),
        ipt_config.get("remove_ratio"),
        ipt_config.get("pos_neg_rat")
    )  
    logging.info("after_shape %s", df.shape)
    if ipt_config.get("change_timestamps", False):
        logging.info("change_timestamps")
        df[oper_time_name] = pd.to_datetime(
            df[oper_time_name],
            format='%Y%m%d%H%M%S', 
            errors='coerce'
        ).astype('int64') // 10**9

    if ipt_config.get("short_oper_date", False):
        df[oper_date_name] = pd.to_datetime(
            df[oper_date_name],
            format='%Y%m%d%H%M%S',
            errors='coerce').dt.strftime('%Y%m%d')

    if ipt_config.get("del_date"):
        for idx in df.index:
            require_idx = get_hist_index(df.loc[idx, hist_date_name], df.loc[idx, oper_date_name])
            for col in history_cols:
                value_list = df.at[idx, col]
                df.at[idx, col] = [value_list[x] if value_list is not None else 0 for x in require_idx] 
    df = df.sort_values(oper_time_name, ascending=False)
    
    first_cols = user_cols + history_cols
    list_cols = cross_cols + [x for x in candidate_cols if x not in group_keys]

    df_merged = optimized_agg_for_multi_groupkeys(
        df,
        group_keys=group_keys,
        first_cols=first_cols,
        list_cols=list_cols
    )
    del df
    gc.collect()
    logging.info("Writing to parquet: %s", opt_file_path)
    if batch_size and batch_size > 0:
        total_rows = len(df_merged)
        total_batches = (total_rows + batch_size - 1) // batch_size  # 向上取整
        # 循环处理每个批次
        for batch_num in range(total_batches):

            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_rows)
            
            # 截取当前批次的DataFrame
            batch_df = df_merged.iloc[start_idx:end_idx]
            
            # 定义输出文件名（包含批次编号）
            output_filename = f"{file_name.replace(file_end,'')}_{batch_num + 1}{opt_file_end}"
            batch_df.to_parquet(os.path.join(output_dir, output_filename), index=False)
    else:
        df_merged.to_parquet(opt_file_path, index=False)
    del df_merged, batch_df
    gc.collect()

    return


def parallel_preprocess(files, ipt_config, num_processes=4):
    """多进程预处理函数"""
    # 准备参数列表
    # 构建进程池并执行任务
    # 使用with语句自动管理进程池的生命周期
    with Pool(processes=num_processes, maxtasksperchild=1) as pool:
        # 准备任务参数（每个文件对应一个参数元组）
        task_args = []

        for file_name in files:
            args_item = {"file_name": file_name}
            args_item.update(ipt_config)
            task_args.append(args_item)
        
        # 并行执行任务
        # imap_unordered: 按任务完成顺序返回结果（效率更高）
        for _ in pool.imap_unordered(process_file, task_args):
            pass
            gc.collect()


def main(_):
    config = read_json(FLAGS.config_file)
    input_dir = config.get("input_dir", FLAGS.INPUT_DIR)
    output_dir = config.get("output_dir", FLAGS.OUTPUT_DIR)

    file_end = config.get("file_end", '.tfrecord.gz')
    opt_file_end = config.get("opt_file_end", ".parquet")

    num_processes = config.get("num_processes", 2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True, mode=0o755)

    files = [f for f in os.listdir(input_dir) if f.endswith(file_end)]
    logging.info("Found %s TFRecord files", len(files))

    group_keys = config.get("group_keys")
    ipt_config = config
    ipt_config.update(
        input_dir=input_dir,
        output_dir=output_dir,
        file_end=file_end,
        history_cols=config.get("history_fields"),  # history类列
        candidate_cols=config.get("candidate_prefix_fields"),
        user_cols=config.get("user_fields"),
        cross_cols=config.get("cross_fields"),
        group_keys=group_keys,
        opt_file_end=opt_file_end,
        label_col=config.get("label_col", "rpk_active_flag"),
        remove_ratio=config.get("remove_ratio", 0.8),
        batch_size=config.get("batch_size", 0),
        del_date=config.get("del_date", False),
        pos_neg_rat=config.get("pos_neg_rat", True),
        col_names=config.get("col_names", {}) 
    )

    parallel_preprocess(
        files=files,
        ipt_config=ipt_config,
        num_processes=num_processes  # 可调整进程数
    )


if __name__ == "__main__":
    app.run(main)
