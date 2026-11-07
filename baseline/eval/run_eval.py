# -*- coding: utf-8 -*-
"""run_eval.py — 汇总评测入口：对若干置信文件统一出指标表。

指标（对齐创新点文档 §3.4）:
  AUROC / ECE15 / Brier / NLL / AURC / SelEX@50,70,90

用法:
  python run_eval.py --labels PATH --conf c1.jsonl c2.jsonl ...
  python run_eval.py --labels PATH --bootstrap A.jsonl B.jsonl --conf A.jsonl B.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import load_labels  # noqa: E402
from common.metrics import summarize, paired_bootstrap_aurc  # noqa: E402


def load_conf(path: str) -> dict[tuple, float]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["qid"], r["source"], r["cand_idx"])] = float(r["conf"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--conf", nargs="+", required=True)
    ap.add_argument("--bootstrap", nargs=2, default=None, metavar=("A", "B"),
                    help="对两个置信文件做 AURC 配对 bootstrap（delta=B-A，<0 说明 B 更好）")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    results: dict[str, tuple[dict[tuple, float], list, list]] = {}
    for path in args.conf:
        conf = load_conf(path)
        keys = [k for k in conf if k in labels]
        if not keys:
            print(f"[warn] {os.path.basename(path)}: 无可用标签样本")
            continue
        scores = [conf[k] for k in keys]
        correct = [labels[k]["correct"] for k in keys]
        results[path] = (dict(zip(keys, scores)), scores, correct)

    header = ["file", "n", "acc", "auroc", "ece15", "brier", "nll", "aurc",
              "sel_ex@50", "sel_ex@70", "sel_ex@90"]
    print("\t".join(header))
    for path, (_, scores, correct) in results.items():
        s = summarize(scores, correct)
        print("\t".join([os.path.basename(path)] +
                        [str(s[k]) for k in header[1:]]))

    if args.bootstrap:
        a_path, b_path = args.bootstrap
        if a_path in results and b_path in results:
            ca, sa, _ = results[a_path]
            cb, sb, _ = results[b_path]
            common = sorted(set(ca) & set(cb))
            if len(common) < 50:
                print(f"[warn] 共同样本不足（{len(common)}），bootstrap 不稳健")
            sa2 = [ca[k] for k in common]
            sb2 = [cb[k] for k in common]
            ok2 = [labels[k]["correct"] for k in common]
            res = paired_bootstrap_aurc(sa2, sb2, ok2, n_boot=args.n_boot)
            print(f"\n[bootstrap AURC delta = {os.path.basename(b_path)} - "
                  f"{os.path.basename(a_path)}] 共同样本 {len(common)}")
            print(f"  delta_mean={res['delta_mean']:.5f} "
                  f"ci95={res['ci95']} p(delta<0)={res['p_delta_lt0']:.4f}")


if __name__ == "__main__":
    main()
