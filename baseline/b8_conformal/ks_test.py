# -*- coding: utf-8 -*-
"""ks_test.py — conformal 交换性假设的正式检验（多 seed）。

背景（Phase 3）：B8 conformal 的 leak 系统性偏离至 ~2α，且超标 seed 与高
coverage 伴生（双峰）。本脚本对每个 seed 除复现 leak 外，做两个正式检验：

  KS-1（两样本）：cal_wrong 与 test_wrong 的 nonconformity 分数（1-conf）
    分布比较。交换性成立 ⇒ 同分布。拒绝 ⇒ 题级聚类导致分布漂移。
  KS-2（单样本）：test_wrong 候选的 conformal p 值 vs U(0,1)。
    H0（候选错误且与 cal 可交换）下 p 均匀；错误候选系统性更自信时 p
    堆积低区（定向拒绝）。

用法:
  python ks_test.py --conf ../../data/pools/bird300_ds_b4.jsonl \
      --labels ../../data/pools/bird300_ds_labels.jsonl --seeds 20
"""
from __future__ import annotations

import argparse
import bisect
import json
import random

from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--cal-frac", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    conf = {}
    with open(args.conf, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                conf[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    float(r["conf"])
    corr = {}
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                k = (int(r["qid"]), str(r["source"]), int(r["cand_idx"]))
                corr[k] = r["label"] == "correct"
    keys = [k for k in conf if k in corr]

    qids = sorted({k[0] for k in keys})
    print(f"# conf={args.conf.split(chr(92))[-1]} n={len(keys)} "
          f"题={len(qids)} alpha={args.alpha} seeds={args.seeds}")
    print("| seed | leak | holds | cov | KS1 D | KS1 p | KS2 D | KS2 p |")
    print("|---|---|---|---|---|---|---|---|")
    n_hold = n_ks1 = n_ks2 = 0
    leaks = []
    for seed in range(args.seeds):
        rq = qids[:]
        random.Random(seed).shuffle(rq)
        n_cal = max(1, int(len(rq) * args.cal_frac))
        cal_q = set(rq[:n_cal])

        cal_w = sorted(1.0 - conf[k] for k in keys
                       if k[0] in cal_q and not corr[k])
        test_w = [1.0 - conf[k] for k in keys
                  if k[0] not in cal_q and not corr[k]]
        test = [(conf[k], corr[k]) for k in keys if k[0] not in cal_q]
        n = len(cal_w)

        accept = [(s, ok) for s, ok in test
                  if (1 + bisect.bisect_right(cal_w, 1.0 - s)) / (n + 1)
                  <= args.alpha]
        acc_wrong = sum(1 for _, ok in accept if not ok)
        leak = acc_wrong / max(1, len(test_w))
        cov = len(accept) / max(1, len(test))
        holds = leak <= args.alpha + 1e-9
        n_hold += holds
        leaks.append(leak)

        ks1 = stats.ks_2samp(cal_w, test_w)
        pvals = [(1 + bisect.bisect_right(cal_w, s)) / (n + 1) for s in test_w]
        ks2 = stats.kstest(pvals, "uniform")
        n_ks1 += ks1.pvalue < 0.05
        n_ks2 += ks2.pvalue < 0.05
        print(f"| {seed} | {leak:.4f} | {holds} | {cov:.4f} "
              f"| {ks1.statistic:.3f} | {ks1.pvalue:.4f} "
              f"| {ks2.statistic:.3f} | {ks2.pvalue:.4f} |")

    leaks.sort()
    over = [l / args.alpha for l in leaks if l > args.alpha]
    print(f"\n# 汇总: 保证成立 {n_hold}/{args.seeds} | "
          f"leak 中位 {leaks[len(leaks)//2]:.4f} | "
          f"超标 seed 的 leak/alpha 比 {[round(x, 2) for x in over]} | "
          f"KS1 拒绝率 {n_ks1}/{args.seeds} | KS2 拒绝率 {n_ks2}/{args.seeds}")


if __name__ == "__main__":
    main()
