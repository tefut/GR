import torch
import torch.distributed as dist

from collections import defaultdict
from typing import List, Dict, Any

from modeling.generic.sequential.features import SequentialFeatures

import logging
import time
import re

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    precision_recall_curve,
    auc
)

from utils.common_utils import weird_division
from utils.common_utils import Const


def _normalize_comparison_operator(value: str) -> str:
    """Convert comparison operators to readable group name suffixes."""
    replacements = {
        '>=': 'gte_',
        '<=': 'lte_',
        '==': 'eq_',
        '>': 'gt_',
        '<': 'lt_'
    }
    for op, suffix in replacements.items():
        value = value.replace(op, suffix)
    return value


class MetricsCalculator:
    def __init__(self, eval_df, featuremap_dict=None, gauc_querys=None, metric_list=None, group_metric_cols=None,
                 bias_evaluation_conf=None, save_result_to_local=True, calculate_group_auc=False):
        self.eval_df = eval_df.copy()
        self.y_true = self.eval_df['ground_truth']
        self.y_score = self.eval_df['scores']

        if self.y_true.isna().sum() > 0:
            logging.info("ground_truth contains NaN, count is %s", self.y_true.isna().sum())
        nan_score_count = self.y_score.isna().sum()
        if nan_score_count > 0:
            logging.warning(
                "score contains NaN, count is %s / %s (%.4f%%), dropping NaN rows for metric calculation",
                nan_score_count, len(self.y_score), 100.0 * nan_score_count / len(self.y_score))
            self.eval_df = self.eval_df.dropna(subset=['scores']).reset_index(drop=True)
            self.y_true = self.eval_df['ground_truth']
            self.y_score = self.eval_df['scores']

        self.metric_list = metric_list
        self.bias_evaluation_conf = bias_evaluation_conf
        self.save_result_to_local = save_result_to_local

        self.feature_map_dict = featuremap_dict
        self.calculate_group_auc = calculate_group_auc

        self.group_metric_cols = gauc_querys

        # 注册指标计算方法
        self.metric_funcs = {
            'auc': self._calc_auc,
            'logloss': self._calc_logloss,
            'bucket_copc': self._calc_bucket_copc,
            'pcoc': self._calc_pcoc,
            'count': self._calc_count,
            'mean': self._calc_mean,
            'variance': self._calc_variance,
            'stddev': self._calc_stddev,
            'segments_copc': self._calc_segments_copc,
            'prauc': self._calc_prauc,
            'ece': self._calc_ece,
            'gauc': self._calc_gauc,
            'gpcoc': self._calc_gpcoc,
            'caln': self._calc_caln,
            'gece': self._calc_gece,
        }

    def calculate(self, output_path):
        # 计算全局指标
        global_results = self._calculate_metrics(self.metric_list, dataset_name="global")

        # 计算分组指标
        logging.info("Starting calculate group metrics")
        group_results = {}
        for group in self.group_metric_cols:
            logging.info(f'Processing group: {group}')
            # 生成分组查询条件
            query_conditions = []
            group_name_parts = []

            for col, values in group.items():
                logging.info(f'Processing column: {col}, values: {values}')
                if isinstance(values, list) and col in self.eval_df.columns:
                    logging.info(f'Column {col} found in eval_df')
                    has_conditions = False
                    if all(isinstance(v, int) for v in values):
                        _values = values
                    else:
                        _values = []

                        for v in values:
                            if isinstance(v, str) and any(op in v for op in ['>=', '<=', '==', '>', '<']):
                                if col in self.eval_df.columns:
                                    # 比较数值大小
                                    condition = f"{col} {v}"
                                    query_conditions.append(condition)
                                    try:
                                        group_name_parts.append(f"{col}_{_normalize_comparison_operator(v)}")
                                    except Exception as e:
                                        logging.warning(f'Failed to normalize comparison operator for {v}: {e}')
                                        group_name_parts.append(f"{col}_{v}")
                                    logging.info(f'Added comparison condition: {condition}')
                                    has_conditions = True
                                else:
                                    logging.info(f'Find Conditional Query, No valid values found for column {col}')
                            else:
                                if col not in self.feature_map_dict:
                                    logging.warning(
                                        f'cannot find column name {col} in values, trying query {values} instead')
                                    result = self.feature_map_dict.get(v, None)
                                else:
                                    result = self.feature_map_dict[col].get(v, None)
                                if result is not None:
                                    _values.append(result)
                                else:
                                    logging.warning(f'cannot find column name {values} in values')

                    if not has_conditions and len(_values) > 0:
                        condition = f"{col} in {_values}" if len(_values) > 1 else f"{col} == {_values[0]}"
                        query_conditions.append(condition)
                        group_name_parts.append(f"{col}={'_'.join(map(str, _values))}")
                        logging.info(f'Added condition: {condition}')
                    elif not has_conditions:
                        logging.info(f'No valid values found for column {col}')
                else:
                    logging.info(
                        'Column %s not in eval_df or values not a list. eval_df columns: %s',
                        col, list(self.eval_df.columns))

            logging.info(f'Final query_conditions: {query_conditions}')
            logging.info(f'Final group_name_parts: {group_name_parts}')

            if query_conditions:
                # 生成分组名称
                group_name = "_AND_".join(group_name_parts)
                logging.info(f'Generated group name: {group_name}')

                # 筛选数据子集
                combined_query = " & ".join(query_conditions)
                logging.info(f'printing query conditions {query_conditions}')
                logging.info(f'printing combined query {combined_query}')

                try:
                    subgroup_df = self.eval_df.query(combined_query, engine='python')
                    logging.info(f'subgroup_df shape: {subgroup_df.shape}')

                    # 子集计算指标
                    if not subgroup_df.empty:
                        logging.info(f'Calculating metrics for subgroup {group_name}')
                        subgroup_calculator = MetricsCalculator(subgroup_df, None, )
                        group_results[group_name] = subgroup_calculator._calculate_metrics(self.metric_list, group_name)
                        logging.info(f'Calculated metrics for {group_name}: {group_results[group_name]}')
                    else:
                        logging.info(f'subgroup_df is empty for {group_name}')
                        group_results[group_name] = {k: None for k in global_results.keys()}
                except Exception as e:
                    logging.error(f'Error occurred while processing group {group_name}: {str(e)}')
                    logging.error(f'Query that failed: {combined_query}')
                    raise e
            else:
                logging.info(f'No valid query conditions for group {group}')

        # 计算偏差评估
        bias_results = {}
        for conf in self.bias_evaluation_conf:
            key = f"{conf['high_level_dimension']}_{conf['low_level_dimension']}"
            bias_results[key] = self._calculate_bias_metrics(
                self.eval_df,
                conf.get('mode', 'all'),
                conf['high_level_dimension'],
                conf['low_level_dimension']
            )

        # 合并结果
        results = {
            "bias_evaluation": bias_results,
            "groups": group_results,
            "global": global_results
        }

        self.print_results(results)
        if self.save_result_to_local:
            self._export_to_csv(results, output_path)

        return results

    def _calculate_metrics(self, metric_list, dataset_name=""):
        """内部计算方法"""
        results = {}
        for metric in metric_list:
            tag = metric['tag']
            start_time = time.time()
            func = self.metric_funcs.get(tag)
            if not func:
                logging.warning("[%s-%s] skip unsupported metric", dataset_name, tag)
                continue

            config = metric.get('config', {})
            try:
                value = func(**config)
                results[metric.get('name', tag)] = round(value, 5) if isinstance(value, float) else value
            except ValueError as e:
                logging.warning("Error: [%s-%s] skip metric: %s", dataset_name, tag, str(e))
                results[metric.get('name', tag)] = None
            except Exception as e:
                logging.warning("[%s-%s] metric calc failed: %s", dataset_name, tag, str(e), exc_info=True)
                results[metric.get('name', tag)] = None
            end_time = time.time()
            logging.info('finished calculation of metrics %s of %s, cost: %.2f s, value: %s', tag, dataset_name,
                         end_time - start_time, results.get(metric.get('name', tag)))
        return results

    def _calculate_bias_metrics(self, df, mode, high_dim, low_dim):
        """引入误差信息和偏差大小进行评估"""
        metrics = {}

        start_time = time.time()

        if mode not in ["all", "mvce", "gc_n"]:
            if not mode:
                mode = 'all'
            else:
                logging.warning("skip unsupported bias_metric: %s", mode)
                return metrics

        # 多维度校准误差
        if mode in ["mvce", "all"]:
            tag = "MVCE"
            try:
                groups = df.groupby(high_dim, observed=False)
                value = groups.apply(self._calculate_pce, bin_col=low_dim).mean()
                metrics[tag] = round(value, 5) if isinstance(value, float) else value
            except ValueError as e:
                logging.warning("Error: [MVCE] skip metric: %s", str(e))
                metrics[tag] = None
            except Exception as e:
                logging.warning("Error: [MVCE] metric calc failed: %s", str(e), exc_info=True)
                metrics[tag] = None
            logging.info('finished calculation of metrics MVCE, value: %s', metrics.get(tag))

        # 分组校准指标
        if mode in ["gc_n", "all"]:
            tag = "GC-N"
            try:
                grouped = df.groupby([high_dim, low_dim], observed=False)
                error_df = grouped.apply(self._calculate_error).reset_index(name='error_i')
                value = error_df.groupby(high_dim, observed=False).apply(self._calculate_gc_n).mean()
                metrics[tag] = round(value, 5) if isinstance(value, float) else value
            except ValueError as e:
                logging.warning("Error: [GC-N] skip metric: %s", str(e))
                metrics[tag] = None
            except Exception as e:
                logging.warning("Error: [GC-N] metric calc failed: %s", str(e), exc_info=True)
                metrics[tag] = None
            logging.info('calculation of metrics GC-N finished, value: %s', metrics.get(tag))
        end_time = time.time()
        logging.info('calculation of bias metrics %s - %s finished, cost: %.2f s', high_dim, low_dim,
                     end_time - start_time)
        return metrics

    def _export_to_csv(self, results, output_path):
        """
        将指标结果导出为CSV文件

        :param: results: calculate()方法返回的结果字典
        :param: output_path: 输出文件路径，如 'metrics_report.csv'
        """
        try:
            records = []
            # 处理全局指标
            global_metrics = results['global']
            logging.info("appending global metrics")
            records.append({
                'scope': 'global',
                'group_name': 'global',
                **global_metrics
            })

            # 处理分组指标
            logging.info("appending metrics by groups")
            for group_name, metrics in results['groups'].items():
                records.append({
                    'scope': 'group',
                    'group_name': group_name,
                    **metrics
                })

            # 将列表转换为dataframe
            logging.info("turn list into dataframe")
            df = pd.DataFrame(records)

            # 调整列顺序
            columns = ['scope', 'group_name'] + [col for col in df.columns if col not in ('scope', 'group_name')]
            df = df[columns]

            df.to_csv(output_path, index=False, float_format='%.5f')
            logging.info("CSV exported, path is : %s, count is: %d", output_path, len(records))
        except PermissionError:
            logging.error("File write permission denied, path is: %s", output_path, exc_info=True)
            raise
        except Exception as e:
            logging.error("Encountered unknown error during CSV export: %s", str(e), exc_info=True)
            raise

    def print_results(self, results, indent=0):
        prefix = " " * indent
        for k, v in results.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                self.print_results(v, indent + 4)
            else:
                print(f"{prefix}{k}: {v}")

    def _calc_auc(self):
        return roc_auc_score(self.y_true, self.y_score)

    def _calc_logloss(self):
        return log_loss(self.y_true, self.y_score)

    def _calc_bucket_copc(self, buckets=10):
        """分桶COPC"""
        eval_df = self.eval_df.loc[:, ['scores', 'ground_truth']]
        try:
            eval_df['bucket'] = pd.qcut(eval_df['scores'], buckets, duplicates='drop')
        except Exception as e:
            logging.error("create buckets failed %s", str(e))
            raise e
        grouped = eval_df.groupby('bucket', observed=False).agg(
            pctr=('scores', 'mean'),
            ctr=('ground_truth', 'mean')
        )
        grouped = grouped[grouped.ctr > 0]
        return (grouped.pctr / grouped.ctr).mean()

    def _calc_pcoc(self):
        """预测点击率与实际点击率比值"""
        pctr = self.y_score.mean()
        ctr = self.y_true.mean()
        return pctr / ctr if ctr != 0 else 0.0

    def _calc_count(self):
        return len(self.y_score)

    def _calc_mean(self):
        return self.y_score.mean()

    def _calc_variance(self):
        return self.y_score.var()

    def _calc_stddev(self):
        return self.y_score.std()

    def _calc_segments_copc(self, copc_pctr_segments):
        """指定分桶点COPC"""
        bins = list(map(float, copc_pctr_segments.split('#')))
        eval_df = self.eval_df.loc[:, ['scores', 'ground_truth']]
        eval_df['bucket'] = pd.cut(eval_df['scores'], bins=bins, include_lowest=True)
        logging.info("buckets created succeed")
        grouped = eval_df.groupby('bucket', observed=False).agg(
            sum_pred=('scores', 'sum'),
            sum_true=('ground_truth', 'sum'),
            count=('scores', 'count')
        )
        logging.info("calculate statistics by group(bucket) finished")

        grouped['copc'] = np.divide(
            grouped['sum_pred'] + Const.EPS,
            grouped['sum_true'] + Const.EPS,
            out=np.zeros_like(grouped['sum_pred']),  # 分母为零时输出0
            where=(grouped['sum_true']) != 0
        )
        return grouped['copc'].tolist()

    def _calc_prauc(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
        return auc(recall, precision)

    def _calc_ece(self, n_bins=10):
        eval_df = self.eval_df.loc[:, ['scores', 'ground_truth']]
        bins = np.linspace(0, 1, n_bins + 1)
        labels = range(n_bins)
        eval_df["bin"] = pd.cut(
            self.y_score, bins=bins, labels=labels, include_lowest=True
        )
        return self._calculate_pce(eval_df, "bin")

    def _gen_group_samples(self, gid, group_sample_rate=1.0):
        group_df = self.eval_df.loc[:, [gid, 'ground_truth', 'scores']]
        group_df[gid] = group_df[gid].astype('category')
        if 0.0 < group_sample_rate < 1.0:
            sampled_gids = group_df[gid].drop_duplicates().sample(frac=group_sample_rate, random_state=42)
            group_df = group_df[group_df[gid].isin(sampled_gids)]
        else:
            logging.info("group_sample_rate must be between 0.0 and 1.0, but got %f", group_sample_rate)
        return group_df.groupby(gid, observed=False)

    def _calc_gauc(self, gid, group_sample_rate=1.0):
        grouped = self._gen_group_samples(gid, group_sample_rate=1.0)
        total_auc = 0.0
        total_weight = 0
        for _, group in grouped:
            y_true = group['ground_truth']
            if y_true.nunique() < 2:
                continue
            total_auc += roc_auc_score(y_true, group['scores']) * len(y_true)
            total_weight += len(y_true)
        return total_auc / total_weight if total_weight > 0 else 0.0

    def _calc_gpcoc(self, gid, group_sample_rate=1.0):
        """分组PCOC"""
        grouped = self._gen_group_samples(gid, group_sample_rate=1.0)
        grouped = grouped.agg(
            pctr=('scores', 'mean'),
            ctr=('ground_truth', 'mean')
        )
        # grouped.ctr为0的数据代表无正样本，在此基础上无法计算pcoc，因此需要过滤
        grouped = grouped[grouped.ctr > 0]
        return (grouped.pctr / grouped.ctr).mean()

    def _calc_caln(self, gid, group_sample_rate=1.0):
        grouped = self._gen_group_samples(gid, group_sample_rate=1.0)
        return np.sqrt(grouped.apply(self._calculate_error).mean(skipna=True))

    def _calc_gece(self, gid, group_sample_rate=1.0):
        group_df = self.eval_df.loc[:, [gid, 'ground_truth', 'scores']]
        group_df[gid] = group_df[gid].astype('category')
        if 0.0 < group_sample_rate < 1.0:
            sampled_gids = group_df[gid].drop_duplicates().sample(frac=group_sample_rate, random_state=42)
            group_df = group_df[group_df[gid].isin(sampled_gids)]
        return self._calculate_pce(group_df, gid)

    def _calculate_pce(self, group, bin_col='bin'):
        pce = 0.0
        total_samples = len(group)

        for _, bin_group in group.groupby(bin_col, observed=False):
            n_b = len(bin_group)
            if n_b == 0:
                continue
            conf_b = bin_group["scores"].mean()
            acc_b = bin_group["ground_truth"].mean()
            pce += (n_b / total_samples) * abs(acc_b - conf_b)
        return pce

    def _calculate_error(self, group):
        sum_gt = group['ground_truth'].sum()
        if sum_gt == 0:
            return np.nan
        sum_scores = group['scores'].sum()
        ratio = weird_division(sum_gt, sum_scores) if sum_gt > sum_scores else weird_division(sum_scores, sum_gt)
        return (ratio - 1) ** 2

    def _calculate_gc_n(self, group):
        mean_squared = group['error_i'].mean()
        return np.sqrt(mean_squared)


@torch.inference_mode
def ag_process_rank_scores(scores, model_input, save_score_csv_item_cols,
                           save_score_csv_user_cols, token_per_item=1):
    """
    使用模型和序列特征评估排名指标。

    :param scores: 模型输出的分数字典。
    :param model_input: 模型输入数据字典
    :param save_score_csv_item_cols: 需要保存到csv的物品级特征列表
    :param save_score_csv_user_cols: 需要保存到csv的用户级特征列表
    :param token_per_item: 每个物品的token数量
    :return: 分数张量、真实标签、group_key以及原始的模型输入
    """
    scores = scores["rerank_score"]

    ground_truth = model_input['labels']  # 0610下candidate_action_type本身只有0曝光1下载，直接可以用做ground_truth

    # 只保留有效位置
    loss_weights = model_input['loss_weights']  # 这里的loss weights直接作为valid test item的筛选，为1则参加计算指标

    # 收集所有需要放进结果 DataFrame 的字段
    eval_group_keys = {
        # 进行广播，将uid扩散到每个位置
        'uid': model_input["uid"].unsqueeze(-1) * torch.ones_like(loss_weights)
    }
    for key in save_score_csv_item_cols:
        if key not in model_input.keys():
            raise ValueError(key + " not in model_inputs")
        eval_group_keys[key] = model_input[key]
    for key in save_score_csv_user_cols:
        if key not in model_input.keys():
            raise ValueError(key + " not in model_inputs")
        eval_group_keys[key] = model_input[key] * torch.ones_like(loss_weights)

    valid_mask = (loss_weights == 1)
    valid_scores = scores[valid_mask]
    valid_labels = ground_truth[valid_mask]

    eval_scores = valid_scores[::token_per_item]
    eval_ground_truth = valid_labels[::token_per_item]
    for key in eval_group_keys.keys():
        eval_group_keys[key] = eval_group_keys[key][valid_mask][::token_per_item]

    return eval_scores, eval_ground_truth, eval_group_keys, model_input


def avg_eval(x: torch.Tensor, world_size: int) -> float:
    """
    计算分布式环境中的平均评估值。

    :param x: 输入的张量。
    :param world_size: 分布式环境中的进程数。
    :return: 平均值。
    """
    _sum_and_numel = torch.tensor([x.sum(), x.numel()], dtype=torch.float32, device=x.device)
    if world_size > 1:
        dist.all_reduce(_sum_and_numel, op=dist.ReduceOp.SUM)
    return torch.div(_sum_and_numel[0], _sum_and_numel[1])


def gather_all_list(x: list, world_size: int) -> torch.Tensor:
    obj_list = [None for _ in range(world_size)]
    dist.all_gather_object(obj_list, x, group=None)
    l = []
    for o in obj_list:
        if isinstance(o, list):
            l.extend(o)
        else:
            l.extend(o.tolist())
    return l


def _merge_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recursively merge a list of dicts:
    - Nested dicts are merged recursively.
    - Lists and tensors/arrays are concatenated.
    - Other values: last one wins.
    """
    merged: Dict[str, Any] = {}
    nested: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seqs: Dict[str, List[Any]] = defaultdict(list)

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                nested[k].append(v)
            elif isinstance(v, list):
                seqs[k].extend(v)
            elif hasattr(v, "tolist"):
                # tensors or numpy arrays
                seqs[k].extend(v.tolist())
            else:
                # scalar or other picklable object
                merged[k] = v

    for k, dict_list in nested.items():
        merged[k] = _merge_dicts(dict_list)

    for k, seq in seqs.items():
        merged[k] = seq

    return merged


def gather_all_dict(x: Dict[str, Any], world_size: int = None) -> Dict[str, Any]:
    """
    Gather a nested dict from all processes and merge into a single dict.
    Automatically strips out callables (e.g., lambdas) before gathering to avoid pickling errors.

    :param x: Local dict to gather.
    :param world_size: Total number of processes. If None, uses dist.get_world_size().
    :return: Merged dict containing data from all processes.
    """

    if world_size is None:
        world_size = dist.get_world_size()

    obj_list: List[Dict[str, Any]] = [None] * world_size
    dist.all_gather_object(obj_list, x, group=None)

    return _merge_dicts(obj_list)
