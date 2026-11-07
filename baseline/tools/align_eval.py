# -*- coding: utf-8 -*-
"""align_eval.py — 多方法同候选集对齐评测（键交集，协议严格可比）。

用法:
  python align_eval.py --labels <labels.jsonl> --pool-prefix <前缀> \
      --methods b1=..b1.jsonl b2=..b2.jsonl ... [--sources nc4] [--out table.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.metrics import auroc, ece, brier, aurc, selective_ex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="name=path 形式，可多对")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="过滤候选来源（如 nc4）；缺省不过滤")
    ap.add_argument("--keep-exec-fail", action="store_true",
                    help="保留 exec_fail 行（默认排除，对齐判官协议）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = {}
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels[(r["qid"], r["source"], r["cand_idx"])] = r["label"]

    ok_labels = {"correct", "silent"} if not args.keep_exec_fail else None
    data: dict[str, dict] = {}
    for spec in args.methods:
        name, path = spec.split("=", 1)
        d = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = (r["qid"], r["source"], r["cand_idx"])
                lab = labels.get(k)
                if lab is None:
                    continue
                if ok_labels and lab not in ok_labels:
                    continue
                if args.sources and k[1] not in args.sources:
                    continue
                d[k] = (float(r["conf"]), lab == "correct")
        data[name] = d

    common = set.intersection(*[set(d) for d in data.values()])
    lines = [f"# 对齐评测 (交集 n={len(common)}"
             + (f", sources={args.sources}" if args.sources else "") + ")",
             "",
             "| method | n | acc | auroc | ece15 | brier | aurc | sel_ex@50 |",
             "|---|---|---|---|---|---|---|---|"]
    for m, d in data.items():
        pairs = [d[k] for k in common]
        sc = [s for s, _ in pairs]
        co = [c for _, c in pairs]
        acc = sum(co) / len(co)
        lines.append(f"| {m} | {len(pairs)} | {acc:.3f} | {auroc(sc, co):.3f} "
                     f"| {ece(sc, co):.3f} | {brier(sc, co):.3f} | {aurc(sc, co):.3f} "
                     f"| {selective_ex(sc, co, 0.5):.3f} |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
