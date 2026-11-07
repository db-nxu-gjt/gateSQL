# -*- coding: utf-8 -*-
"""B2 Consistency（Self-Consistency 投票）— 黑盒多次采样置信。

出处: Maleki et al., AAAI 2026 (arXiv:2508.14056) 中黑盒最优之一；
      Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"。
协议: 对齐创新点文档 §3.2 —— 所有方法评同一批候选池。对每题取同一
      (qid, source) 的 k 个候选，各自执行后按严格结果集聚类，
      conf = 多数派占比（agreement share）。

输入: 候选池 jsonl（>=k 候选/题；可直接用 AUC 池的 nc4 来源 k=4，
      或 make_pool.py 生成的 nucleus10 来源 k=10）
输出: confidence jsonl（同时可补写执行标注 labels）

用法:
  python run_b2.py --pool PATH --out PATH [--k 5] [--source nc4]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import Candidate, load_pool, write_jsonl  # noqa: E402
from common import exec_match  # noqa: E402


def group_questions(pool: list[Candidate], source: str | None):
    groups: dict[tuple, list[Candidate]] = defaultdict(list)
    for c in pool:
        if source and c.source != source:
            continue
        groups[(c.qid, c.source)].append(c)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None,
                    help="可选：把执行标注补写到该 labels.jsonl")
    ap.add_argument("--k", type=int, default=5,
                    help="每题取 k 个候选投票（默认 5）")
    ap.add_argument("--source", default=None, help="只用指定来源的候选")
    ap.add_argument("--dataset", default="bird", choices=["bird", "spider"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    groups = group_questions(pool, args.source)
    print(f"[b2] {len(groups)} 题, k={args.k}")

    # 批量执行标注（复用 AUC labels 若给出，否则现场执行并缓存）
    all_cands = [c for g in groups.values() for c in g]
    labels = exec_match.label_pool(all_cands, args.dataset, cache_path=args.labels)
    print(f"[b2] 标注完成: {len(labels)} 条")

    rng = random.Random(args.seed)
    rows = []
    for (qid, src), cands in sorted(groups.items()):
        picked = cands if len(cands) <= args.k else rng.sample(cands, args.k)
        results = [labels.get(c.key) for c in picked]
        valid = [r for r in results if r == "correct" or r == "silent"]
        if not valid:  # 全部执行失败 -> 无信息，给中性分
            conf = 0.0
        else:
            majority = max(valid.count("correct"), valid.count("silent"))
            conf = majority / len(valid)
        for c in picked:
            rows.append({"qid": c.qid, "source": c.source,
                         "cand_idx": c.cand_idx,
                         "conf": conf, "method": f"b2_consistency_k{args.k}"})

    write_jsonl(args.out, rows)
    print(f"[done] {len(rows)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
