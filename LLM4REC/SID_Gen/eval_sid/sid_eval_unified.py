# -*- coding: utf-8 -*-
"""
SID Evaluator
评估 SID 生成结果的质量

输入：
- parquet 文件
- 必须包含: 主键, SID 列
- 可选包含类目列

配置：
- 通过 --config 指定 YAML 配置文件
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import normalized_mutual_info_score

SID_RE = re.compile(r"<\s*([a-zA-Z])\s*[_-]\s*(\d+)\s*>")


def parse_sid_to_codes(s: str, depth: int) -> List[int]:
    """
    输入:
        "<a_187><b_214><c_187>"
    输出:
        [187, 214, 187]
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        raise ValueError("sid is null")

    s = str(s).strip()
    matches = SID_RE.findall(s)

    if not matches:
        raise ValueError("Cannot parse sid: %s" % s)

    codes = [int(num) for _, num in matches]

    if len(codes) < depth:
        codes = codes + [-1] * (depth - len(codes))
    else:
        codes = codes[:depth]

    return codes


def entropy_bits_from_counts(counts_all: np.ndarray) -> float:
    total = counts_all.sum()
    if total <= 0:
        return 0.0
    p = counts_all[counts_all > 0] / total
    return float(-np.sum(p * np.log2(p)))


def gini_from_counts(counts_all: np.ndarray) -> float:
    sorted_counts = np.sort(counts_all.astype(np.float64))
    total = sorted_counts.sum()
    if total <= 0:
        return 0.0

    k = len(sorted_counts)
    idx = np.arange(1, k + 1)
    return (2 * np.sum(idx * sorted_counts)) / (k * total) - (k + 1) / k


class FinalSIDEvaluator:
    def __init__(
            self,
            input_path: str,
            id_col: str,
            sid_col: str,
            name_col: str,
            cat_cols: Optional[List[str]],
            depth: int,
            vocab_sizes: List[int],
            logger: logging.Logger,
    ):
        self.input_path = input_path
        self.id_col = id_col
        self.sid_col = sid_col
        self.cat_cols = cat_cols or []
        self.depth = depth
        self.vocab_sizes = vocab_sizes
        self.logger = logger
        self.sid2names = defaultdict(list)
        self.name_col = name_col

        if len(self.vocab_sizes) == 1:
            self.vocab_sizes = self.vocab_sizes * self.depth
        if len(self.vocab_sizes) != self.depth:
            raise ValueError("VOCAB_SIZES length must equal DEPTH")

        self.df = self._load_and_prepare()

    def _load_and_prepare(self) -> pd.DataFrame:
        self.logger.info("Loading parquet: %s", self.input_path)
        df = pd.read_parquet(self.input_path)

        if self.id_col not in df.columns:
            raise ValueError("Missing id_col: %s" % self.id_col)
        if self.sid_col not in df.columns:
            raise ValueError("Missing sid_col: %s" % self.sid_col)

        df = df.drop_duplicates(subset=[self.id_col], keep="first").copy()

        valid_cat_cols = [c for c in self.cat_cols if c in df.columns]
        self.cat_cols = valid_cat_cols

        parsed_rows = []
        bad = 0

        use_cols = [self.id_col, self.sid_col] + self.cat_cols
        if self.name_col:
            use_cols = use_cols + [self.name_col]
        for _, row in df[use_cols].iterrows():
            try:
                codes = parse_sid_to_codes(row[self.sid_col], self.depth)
                if self.name_col:
                    self.sid2names[tuple(codes)].append(row[self.name_col])
                item = {
                    self.id_col: row[self.id_col],
                    self.sid_col: row[self.sid_col],
                }
                for i in range(self.depth):
                    item[f"Code_L{i + 1}"] = codes[i]
                for c in self.cat_cols:
                    item[c] = row[c]
                parsed_rows.append(item)
            except Exception:
                bad += 1

        out_df = pd.DataFrame(parsed_rows)

        if bad > 0:
            self.logger.info("Skipped %d rows due to SID parse error.", bad)

        self.logger.info("Parsed %d rows, depth=%d", len(out_df), self.depth)

        self.logger.info("Preview parsed top 10:")
        preview_cols = [self.id_col, self.sid_col] + [f"Code_L{i + 1}" for i in range(self.depth)] + self.cat_cols
        print("\n========== PARSED PREVIEW (TOP 10) ==========\n")
        print(out_df[preview_cols].head(10).to_string(index=False))
        print("\n=============================================\n")

        return out_df

    def run(self):
        self._section("1) 码本容量")
        self.eval_capacity()

        self._section("2) 最终 SID 冲突率")
        self.eval_collision()

        self._section("2.1) SID 重复明细")
        self.eval_sid_duplicates(topk=20)

        self._section("3) 各层利用率 / 熵 / GINI")
        self.eval_layers()

        self._section("4) Prefix Collision（L1 / L1-L2）")
        self.eval_prefix_collision()

        self._section("5) Prefix Duplicates（L1 / L1-L2）")
        self.eval_prefix_duplicates()

        self._section("6) NMI（Code_L1 vs 类目）")
        self.eval_nmi()

    def eval_capacity(self):
        cap = int(np.prod(self.vocab_sizes))
        self.logger.info("Vocab sizes: %s", self.vocab_sizes)
        self.logger.info("Codebook capacity: %d", cap)

    def eval_collision(self):
        tuples = list(
            zip(*[self.df[f"Code_L{i + 1}"].tolist() for i in range(self.depth)])
        )
        n = len(tuples)
        uniq = len(set(tuples))
        collisions = n - uniq

        self.logger.info(
            "Collision Rate: %.6f (%d collisions / %d)",
            collisions / max(n, 1),
            collisions,
            n,
        )
        self.logger.info("Unique Semantic Paths: %d", uniq)

    def eval_sid_duplicates(self, topk: int = 20):
        tuples = list(
            zip(*[self.df[f"Code_L{i + 1}"].tolist() for i in range(self.depth)])
        )
        counter = Counter(tuples)

        dup = [(sid, cnt) for sid, cnt in counter.items() if cnt > 1]
        if not dup:
            self.logger.info("No duplicated SID tuples found.")
            return

        dup.sort(key=lambda x: x[1], reverse=True)
        total_dup_items = sum(cnt for _, cnt in dup)

        self.logger.info(
            "Duplicated SID tuples: %d types, covering %d items (%.2f%%)",
            len(dup),
            total_dup_items,
            total_dup_items / max(len(tuples), 1) * 100,
        )

        for i, (sid, cnt) in enumerate(dup[:topk], 1):
            self.logger.info("#%-2d SID=%s count=%d names:%s", i, sid, cnt, str(self.sid2names.get(sid, [])[:10]))

    def eval_layers(self):
        self.logger.info("%-6s | %-8s | %-10s | %-10s | %-10s", "Layer", "Vocab", "Util(%)", "Entropy", "Gini")
        self.logger.info("-" * 60)

        for d in range(self.depth):
            k = self.vocab_sizes[d]
            col = f"Code_L{d + 1}"

            layer = self.df[col].to_numpy()
            layer = layer[layer >= 0]
            counts = Counter(layer.tolist())

            counts_all = np.zeros(k, dtype=np.float64)
            for code, cnt in counts.items():
                if 0 <= code < k:
                    counts_all[code] = cnt

            util = np.count_nonzero(counts_all) / max(k, 1)
            ent = entropy_bits_from_counts(counts_all)
            gini = gini_from_counts(counts_all)

            self.logger.info(
                "L%-5d | %-8d | %-10.2f | %-10.4f | %-10.4f",
                d + 1,
                k,
                util * 100,
                ent,
                gini,
            )

    def eval_prefix_collision(self):
        n = len(self.df)

        l1 = list(self.df["Code_L1"].tolist())
        uniq_l1 = len(set(l1))
        col_l1 = n - uniq_l1
        self.logger.info(
            "L1 Collision Rate: %.6f (%d collisions / %d)",
            col_l1 / max(n, 1),
            col_l1,
            n,
        )

        if self.depth >= 2:
            l12 = list(zip(self.df["Code_L1"].tolist(), self.df["Code_L2"].tolist()))
            uniq_l12 = len(set(l12))
            col_l12 = n - uniq_l12
            self.logger.info(
                "L1-L2 Prefix Collision Rate: %.6f (%d collisions / %d)",
                col_l12 / max(n, 1),
                col_l12,
                n,
            )

    def eval_prefix_duplicates(self):
        l1 = self.df[f"Code_L1"].tolist()
        counter = Counter(l1)

        dup = [(sid, cnt) for sid, cnt in counter.items() if cnt > 1]
        total_dup_items = sum(cnt for _, cnt in dup)

        self.logger.info(
            "L1 Duplicated SID tuples: %d types, covering %d items (%.2f%%)",
            len(dup),
            total_dup_items,
            total_dup_items / max(len(l1), 1) * 100,
        )
        if self.depth >= 2:
            tuples = list(
                zip(*[self.df[f"Code_L{i + 1}"].tolist() for i in range(2)])
            )
            counter = Counter(tuples)
            dup = [(sid, cnt) for sid, cnt in counter.items() if cnt > 1]
            total_dup_items = sum(cnt for _, cnt in dup)
            self.logger.info(
                "L1-L2 Duplicated SID tuples: %d types, covering %d items (%.2f%%)",
                len(dup),
                total_dup_items,
                total_dup_items / max(len(tuples), 1) * 100,
            )

    def eval_nmi(self):
        if not self.cat_cols:
            self.logger.info("No CAT_COLS provided. Skip NMI.")
            return

        y = self.df["Code_L1"].fillna(-1).astype(str)

        for c in self.cat_cols:
            x = self.df[c].fillna("Unknown").astype(str)
            score = normalized_mutual_info_score(x, y)
            self.logger.info("NMI(%s, Code_L1)=%.6f", c, score)

    def _section(self, title: str):
        self.logger.info("\n%s\n%s\n%s", "=" * 60, title, "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SID Evaluation")

    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (YAML格式)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    input_parquet = paths.get("input_parquet", "")
    base_dir = paths.get("base_dir", "")

    if base_dir and not os.path.isabs(input_parquet):
        input_parquet = os.path.join(base_dir, input_parquet)

    eval_config = cfg.get("eval", {})
    sid_col = eval_config.get("sid_col", "sid")
    name_col = eval_config.get("name_col", "")
    depth = eval_config.get("depth", 3)
    vocab_sizes = eval_config.get("vocab_sizes", [256, 256, 256])
    cat_cols = eval_config.get("cat_cols", [])

    primary_key = cfg.get("primary_key", "app_id")

    logging_cfg = cfg.get("logging", {})
    log_level = logging_cfg.get("level", "INFO")
    log_format = logging_cfg.get("format", "%(asctime)s [%(levelname)s] %(message)s")
    log_file = logging_cfg.get("log_file", "")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format
    )
    logger = logging.getLogger("sid_eval")

    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)

    evaluator = FinalSIDEvaluator(
        input_path=input_parquet,
        id_col=primary_key,
        sid_col=sid_col,
        name_col=name_col,
        cat_cols=cat_cols,
        depth=depth,
        vocab_sizes=vocab_sizes,
        logger=logger,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
