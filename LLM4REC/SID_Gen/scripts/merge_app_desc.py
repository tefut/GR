# -*- coding: utf-8 -*-

"""
按 app_id 将第二个 txt 的 app_desc 合并到第一个 txt（不丢行，带坏行修复）

修复目标：
- 不丢任何行
- 尽量保证 app_id、app_cn_name 两列正确
- 对于列数异常（通常是字段里含 \t），将“多出来的列碎片”吞进 app_cn_name，
  再用行尾对齐保证后续字段尽可能正确

输入默认 TSV（tab 分隔），带表头
"""

# Standard Library Imports
import csv
import json
import os
from typing import Dict, Any, List, Tuple

# Custom Module Imports
from log_utils import init_logger

CONFIG_FILE = "train.config"


def load_config(config_path: str, logger) -> Dict[str, Any]:
    """
    读取配置文件（JSON）
    必填：input_txt_1 / input_txt_2 / output_txt
    可选：badcase_txt（默认 output_txt + ".badcase.txt"）
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError("config file not found: %s" % config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required = ["input_txt_1", "input_txt_2", "output_txt"]
    for key in required:
        if key not in config or not str(config[key]).strip():
            raise ValueError("missing config key: %s" % key)

    if "badcase_txt" not in config or not str(config["badcase_txt"]).strip():
        config["badcase_txt"] = str(config["output_txt"]) + ".badcase.txt"

    logger.info("config loaded from %s", config_path)
    return config


def inspect_header(file_path: str, logger, delimiter: str = "\t") -> None:
    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            header_line = f.readline()
    except Exception as e:
        logger.warning("failed to read header for inspection: %s | err=%s", file_path, str(e))
        return

    if header_line == "":
        logger.warning("file is empty, cannot inspect header: %s", file_path)
        return

    header_line_strip = header_line.rstrip("\n")
    cols = header_line_strip.split(delimiter)

    empty_name_cnt = 0
    for c in cols:
        if c is None or str(c).strip() == "":
            empty_name_cnt += 1

    logger.info("header inspect file=%s", file_path)
    logger.info("header raw repr=%s", repr(header_line_strip))
    logger.info("header col count=%d", len(cols))
    logger.info("header empty-name col count=%d", empty_name_cnt)

    if header_line_strip.endswith(delimiter):
        logger.warning("header ends with delimiter '%s' -> likely extra empty column at end", delimiter)
    if (delimiter + delimiter) in header_line_strip:
        logger.warning("header contains consecutive delimiters '%s%s' -> likely empty column in middle", \
                       delimiter, delimiter)


def build_desc_map(txt_path: str, logger, delimiter: str = "\t") -> Dict[str, str]:
    """
    构建 app_id -> app_desc
    """
    logger.info("start loading desc map: %s", txt_path)

    desc_map: Dict[str, str] = {}

    with open(txt_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if not reader.fieldnames:
            raise ValueError("empty header in %s" % txt_path)

        logger.info("desc file fieldnames=%s", reader.fieldnames)

        if "app_id" not in reader.fieldnames:
            raise ValueError("column 'app_id' not found in %s" % txt_path)
        if "app_desc" not in reader.fieldnames:
            raise ValueError("column 'app_desc' not found in %s" % txt_path)

        for i, row in enumerate(reader, 1):
            app_id = (row.get("app_id") or "").strip()
            if not app_id:
                continue
            desc_map[app_id] = (row.get("app_desc") or "").strip()

            if i % 10000 == 0:
                logger.info("loaded %d rows from desc file", i)

    logger.info("desc map size = %d", len(desc_map))
    return desc_map


def _clean_fieldnames(raw_fieldnames: List[str]) -> List[str]:
    cleaned = []
    for fn in raw_fieldnames:
        if fn is None:
            continue
        if str(fn).strip() == "":
            continue
        cleaned.append(fn)
    return cleaned


def repair_cols_keep_appid_appname(
        cols: List[str],
        raw_fieldnames: List[str],
        cleaned_fieldnames: List[str],
        logger,
        line_no: int,
        delimiter: str = "\t",
        absorb_col: str = "app_cn_name",
) -> Tuple[Dict[str, str], str]:
    """
    把 split 出来的 cols 修复成 {field: value}：
    - 如果 cols 数量 > header 列数：把多出来的列吞进 app_cn_name，尾部对齐其余字段
    - 如果 cols 数量 < header 列数：末尾补空
    返回：(row_dict, reason)
    """
    n_header = len(raw_fieldnames)
    n_cols = len(cols)

    # 先做最保守的补齐/截断（为了避免崩）
    reason = "ok"
    if n_cols == n_header:
        reason = "ok"
    elif n_cols < n_header:
        reason = "short_cols"
        cols = cols + [""] * (n_header - n_cols)
    else:
        # n_cols > n_header：关键修复逻辑
        reason = "extra_cols"

        # 字段位置（按你 header 定义）
        # 0 app_id
        # 1 dev_up_id
        # 2 app_first_type
        # 3 app_second_type
        # 4 app_third_type
        # 5 app_cn_name  <-- 吞 extra
        # 6 package_name
        # ...
        # 尾部对齐：从 package_name(索引6) 到最后，一共有 (n_header - 6) 个字段
        tail_count = n_header - 6
        if tail_count < 0:
            tail_count = 0

        prefix = cols[0:5]  # 固定前5列
        tail = cols[-tail_count:] if tail_count > 0 else []

        middle = cols[5: len(cols) - tail_count] if tail_count > 0 else cols[5:]
        # 将 middle 拼回 app_cn_name（用空格拼接，避免再次引入 tab）
        app_name_fixed = " ".join([x for x in middle if x is not None and str(x) != ""])

        cols = prefix + [app_name_fixed] + tail

        # 保险：修复后仍可能不等长（极端情况），再兜底
        if len(cols) < n_header:
            cols = cols + [""] * (n_header - len(cols))
        elif len(cols) > n_header:
            cols = cols[:n_header]

        if line_no <= 5:
            logger.warning("sample repaired(extra_cols) line_no=%d old_cols=%d new_cols=%d app_id=%s",
                           line_no, n_cols, len(cols), prefix[0] if prefix else "")

    # 映射到 dict（按 raw_fieldnames）
    row_raw = {}
    for i, fn in enumerate(raw_fieldnames):
        row_raw[fn] = cols[i] if i < len(cols) else ""

    # 清理 key=None（如 header 有空列名）
    if None in row_raw:
        row_raw.pop(None, None)
        reason = "none_key"

    # 构建输出 row：只保留 cleaned_fieldnames
    row = {}
    for fn in cleaned_fieldnames:
        row[fn] = row_raw.get(fn, "")

    return row, reason


def merge_file(
        input_file: str,
        output_file: str,
        badcase_file: str,
        desc_map: Dict[str, str],
        logger,
        delimiter: str = "\t",
):
    """
    合并 app_desc，并将异常原始行写入 badcase 文件（但不丢：修复后仍写入 output）
    """

    logger.info("start merging")
    logger.info("input file  : %s", input_file)
    logger.info("output file : %s", output_file)
    logger.info("badcase file: %s", badcase_file)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(badcase_file) or ".", exist_ok=True)

    # 读取 header（raw_fieldnames）
    with open(input_file, "r", encoding="utf-8", newline="") as fin:
        header_line = fin.readline()
        if header_line == "":
            raise ValueError("input file is empty: %s" % input_file)

        raw_fieldnames = header_line.rstrip("\n").split(delimiter)
        cleaned_fieldnames = _clean_fieldnames(raw_fieldnames)

    if "app_id" not in cleaned_fieldnames:
        raise ValueError("column 'app_id' not found in %s" % input_file)

    if "app_desc" not in cleaned_fieldnames:
        cleaned_fieldnames.append("app_desc")

    logger.info("input raw_fieldnames=%s", raw_fieldnames)
    logger.info("input cleaned fieldnames=%s", cleaned_fieldnames)

    total_lines = 0  # 输入行数（不含 header）
    total_written = 0  # 写入 output 的行数（应等于 total_lines）
    hit = 0  # 命中 app_desc 的行数
    repaired_extra = 0  # extra_cols 修复次数
    repaired_short = 0  # short_cols 修复次数
    repaired_none = 0  # none_key 修复次数

    with open(input_file, "r", encoding="utf-8", newline="") as fin, \
            open(output_file, "w", encoding="utf-8", newline="") as fout, \
            open(badcase_file, "w", encoding="utf-8") as fbad:

        # 跳过 header
        _ = fin.readline()

        writer = csv.DictWriter(
            fout,
            fieldnames=cleaned_fieldnames,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()

        # badcase header
        fbad.write("# badcase lines from: %s\n" % input_file)
        fbad.write("# raw_header: %s\n" % header_line.rstrip("\n"))
        fbad.write("# format: reason | line_no | old_cols | header_cols | app_id | app_cn_name | raw_line\n")

        header_cols = len(raw_fieldnames)

        line_no = 1  # 从 1 开始计数（不含 header）
        while True:
            raw_line = fin.readline()
            if raw_line == "":
                break

            total_lines += 1
            raw_line_strip = raw_line.rstrip("\n")

            # split（这里不使用 csv.reader，因为你的文件并没有标准引号转义）
            cols = raw_line_strip.split(delimiter)

            row, reason = repair_cols_keep_appid_appname(
                cols=cols,
                raw_fieldnames=raw_fieldnames,
                cleaned_fieldnames=cleaned_fieldnames,
                logger=logger,
                line_no=line_no,
                delimiter=delimiter,
                absorb_col="app_cn_name",
            )

            if reason == "extra_cols":
                repaired_extra += 1
                app_id = (row.get("app_id") or "").strip()
                app_cn_name = (row.get("app_cn_name") or "").strip()
                fbad.write("%s | %d | %d | %d | %s | %s | %s\n" %
                           (reason, line_no, len(cols), header_cols, app_id, app_cn_name, raw_line_strip))
            elif reason == "short_cols":
                repaired_short += 1
                app_id = (row.get("app_id") or "").strip()
                app_cn_name = (row.get("app_cn_name") or "").strip()
                fbad.write("%s | %d | %d | %d | %s | %s | %s\n" %
                           (reason, line_no, len(cols), header_cols, app_id, app_cn_name, raw_line_strip))
            elif reason == "none_key":
                repaired_none += 1
                app_id = (row.get("app_id") or "").strip()
                app_cn_name = (row.get("app_cn_name") or "").strip()
                fbad.write("%s | %d | %d | %d | %s | %s | %s\n" %
                           (reason, line_no, len(cols), header_cols, app_id, app_cn_name, raw_line_strip))

            # 合并 app_desc
            app_id = (row.get("app_id") or "").strip()
            desc = desc_map.get(app_id, "")
            if desc:
                hit += 1
            row["app_desc"] = desc

            writer.writerow(row)
            total_written += 1

            if total_lines % 100000 == 0:
                logger.info("processed %d rows, matched %d rows", total_lines, hit)

            line_no += 1

    logger.info("merge finished")
    logger.info("input rows            : %d", total_lines)
    logger.info("output rows           : %d", total_written)
    logger.info("matched rows          : %d", hit)
    logger.info("repaired extra_cols   : %d", repaired_extra)
    logger.info("repaired short_cols   : %d", repaired_short)
    logger.info("repaired none_key     : %d", repaired_none)
    logger.info("badcase file saved to : %s", badcase_file)


def main():
    logger = init_logger(
        name="merge_app_desc",
        log_file="merge_app_desc.log",
        console=True,
    )

    logger.info("program start")

    config = load_config(CONFIG_FILE, logger)

    input_txt_1 = config["input_txt_1"]
    input_txt_2 = config["input_txt_2"]
    output_txt = config["output_txt"]
    badcase_txt = config["badcase_txt"]

    logger.info("input_txt_1 = %s", input_txt_1)
    logger.info("input_txt_2 = %s", input_txt_2)
    logger.info("output_txt  = %s", output_txt)
    logger.info("badcase_txt = %s", badcase_txt)

    inspect_header(input_txt_1, logger, delimiter="\t")
    inspect_header(input_txt_2, logger, delimiter="\t")

    desc_map = build_desc_map(input_txt_2, logger, delimiter="\t")

    merge_file(
        input_file=input_txt_1,
        output_file=output_txt,
        badcase_file=badcase_txt,
        desc_map=desc_map,
        logger=logger,
        delimiter="\t",
    )

    logger.info("program finished")


if __name__ == "__main__":
    main()
