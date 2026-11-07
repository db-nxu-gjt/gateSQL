# -*- coding: utf-8 -*-
"""run_fuse.py — 多信号融合分数（B4 白盒置信 × 执行一致性证据）。

动机（Phase 3 结论）：conformal 自动执行区要求"错误率≈0 的最高分数区"，
单一置信分数（AUROC 0.647）无法提供。本脚本做两级栈式融合：
  L1: B4 子句频率逻辑回归（复用 run_b4_recalib 的特征与拟合器；
      fit 集 2 折 OOF 得无偏分数，全 fit 模型给 eval 分数）
  L2: [logit(OOF b4), 一致性簇占比, 交互项] 逻辑回归 → 融合分数
一致性证据 = 同 (qid, source) 采样组内执行结果多数派占比（B2 协议，
nucleus10 / beam10 各自组内投票）。

fit/eval 划分与 run_b4_recalib 完全一致（同 seed 同题集），
输出仅含 eval 划分，保证与 B4/B8 现有结果同集可比。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
from common.schema import load_pool, load_labels, write_jsonl  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--shares", nargs="+", required=True,
                    help="一致性簇占比文件（b2 nucleus10/beam10 等）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool, require_logprobs=True)
    labels = load_labels(args.labels)
    data = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue  # 与 B4/B8 协议一致：保留 exec_fail（标签 False）
        z = sum(c.token_logprobs) if c.token_logprobs else 0.0
        data.append((c, z, clause_features(c.sql), lab["correct"]))
    print(f"[fuse] 候选 {len(data)} (含 exec_fail，与 B4/B8 协议一致)")

    # ---- 一致性簇占比证据（多个来源文件合并；缺失用 0.5 中性） ----
    shares: dict = {}
    for f in args.shares:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                shares[(r["qid"], r["source"], r["cand_idx"])] = r["conf"]
    print(f"[fuse] share 证据 {len(shares)} 条")

    # ---- L1: B4 管线（与 run_b4_recalib 同划分同 seed） ----
    qids = sorted({c.qid for c, _, _, _ in data})
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    n_fit = max(1, int(len(qids) * args.fit_frac))
    fit_qids = set(qids[:n_fit])
    fit_idx = [i for i, d in enumerate(data) if d[0].qid in fit_qids]
    eva_idx = [i for i, d in enumerate(data) if d[0].qid not in fit_qids]

    dim = len(data[0][2])
    mu = [sum(data[i][2][j] for i in fit_idx) / len(fit_idx) for j in range(dim)]
    sd = [max(1e-6, math.sqrt(sum((data[i][2][j] - mu[j]) ** 2
                                  for i in fit_idx) / len(fit_idx)))
          for j in range(dim)]

    def make_x(i):
        return ([(fj - m) / s for fj, m, s in zip(data[i][2], mu, sd)]
                + [data[i][1] / 10.0])

    def raw_score(i, params_):
        x = make_x(i)
        zz = params_[-1] + sum(w * xj for w, xj in zip(params_[:-1], x))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

    # fit 内 2 折 OOF（L2 的无偏输入）
    fq = sorted(fit_qids)
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(fq)
    half = max(1, len(fq) // 2)
    folds = (set(fq[:half]), set(fq[half:]))
    oof = {}  # i -> score
    for tr, te in ((folds[0], folds[1]), (folds[1], folds[0])):
        tr_idx = [i for i in fit_idx if data[i][0].qid in tr]
        te_idx = [i for i in fit_idx if data[i][0].qid in te]
        p_ = fit_logistic([make_x(i) for i in tr_idx],
                          [data[i][3] for i in tr_idx])
        for i in te_idx:
            oof[i] = raw_score(i, p_)
    params_l1 = fit_logistic([make_x(i) for i in fit_idx],
                             [data[i][3] for i in fit_idx])

    def share_of(i):
        return shares.get(data[i][0].key, 0.5)

    # ---- L2: [logit(b4), share, 交互] 逻辑回归 ----
    def l2_x(s: float, sh: float):
        lg = logit(s)
        return [lg, sh, lg * sh]

    l2_fit = [i for i in fit_idx if i in oof]
    l2_raw = [l2_x(oof[i], share_of(i)) for i in l2_fit]
    l2_y = [1.0 if data[i][3] else 0.0 for i in l2_fit]
    d2 = len(l2_raw[0])
    mu2 = [sum(r[j] for r in l2_raw) / len(l2_raw) for j in range(d2)]
    sd2 = [max(1e-6, math.sqrt(sum((r[j] - mu2[j]) ** 2
                                   for r in l2_raw) / len(l2_raw)))
           for j in range(d2)]
    params_l2 = fit_logistic(
        [[(r[j] - mu2[j]) / sd2[j] for j in range(d2)] for r in l2_raw], l2_y)

    rows = []
    for i in eva_idx:
        c = data[i][0]
        x = l2_x(raw_score(i, params_l1), share_of(i))
        zz = params_l2[-1] + sum(w * xj for w, xj in
                                 zip(params_l2[:-1],
                                     [(xj - m) / s for xj, m, s in zip(x, mu2, sd2)]))
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))
        rows.append({"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                     "conf": p, "method": "fuse_b4xconsist"})
    write_jsonl(args.out, rows)
    print(f"[done] fit={len(l2_fit)} eval={len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
