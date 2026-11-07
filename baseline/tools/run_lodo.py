# -*- coding: utf-8 -*-
"""run_lodo.py — C1: leave-one-db-out 跨库泛化（RQ4）。

对照设计：
  全局 i.i.d.（现有 B4/fuse 结果）：fit/eval 按**题**划分，fit 与 eval 混库；
  LODO（本脚本）：按**库**划分，每折 held-out 一个库，在其余库拟合
  （B4 逻辑回归 + fuse L2），在 held-out 库评测。B1/B3 无拟合，直接切分。

报告每折 n/AUROC/ECE + macro 平均。与全局的差距 = 跨库泛化损失
（排序损失看 AUROC 差，校准漂移看 ECE 差）。

用法:
  python run_lodo.py --pool ../../data/pools/bird300_ds_pool.jsonl \
      --labels ../../data/pools/bird300_ds_labels.jsonl \
      --shares ../../data/pools/bird300_ds_b2_nucleus10.jsonl \
               ../../data/pools/bird300_ds_b2_beam10.jsonl \
      --b1 ../../data/pools/bird300_ds_b1.jsonl \
      --b3 ../../data/pools/bird300_ds_b3_sac.jsonl
"""
from __future__ import annotations

import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
from common.schema import load_pool, load_labels  # noqa: E402
from common import metrics as M  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--shares", nargs="+", required=True)
    ap.add_argument("--b1", required=True)
    ap.add_argument("--b3", required=True)
    args = ap.parse_args()

    pool = load_pool(args.pool, require_logprobs=True)
    labels = load_labels(args.labels)
    data = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue
        z = sum(c.token_logprobs) if c.token_logprobs else 0.0
        data.append((c, z, clause_features(c.sql), lab["correct"]))

    shares: dict = {}
    for f in args.shares:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    shares[(r["qid"], r["source"], r["cand_idx"])] = r["conf"]

    ext: dict = {}  # tag -> {(qid,source,idx): conf}
    for tag, f in (("b1", args.b1), ("b3", args.b3)):
        m = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    m[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                        float(r["conf"])
        ext[tag] = m

    dbs = sorted({d[0].db_id for d in data})
    print(f"# LODO: {len(data)} 候选 / {len(dbs)} 库: {', '.join(dbs)}")
    print("| fold(db) | n | b1 AUROC | b3 AUROC | b4-lo AUROC | fuse-lo AUROC "
          "| b1 ECE | b3 ECE | b4-lo ECE | fuse-lo ECE |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    agg: dict = {}
    for db in dbs:
        te = [i for i, d in enumerate(data) if d[0].db_id == db]
        tr = [i for i, d in enumerate(data) if d[0].db_id != db]

        # B4 LODO 拟合（train 库；标准化/模型均不接触 test 库）
        dim = len(data[0][2])
        mu = [sum(data[i][2][j] for i in tr) / len(tr) for j in range(dim)]
        sd = [max(1e-6, math.sqrt(sum((data[i][2][j] - mu[j]) ** 2
                                      for i in tr) / len(tr)))
              for j in range(dim)]

        def make_x(i):
            return ([(fj - m) / s for fj, m, s in zip(data[i][2], mu, sd)]
                    + [data[i][1] / 10.0])

        def raw_score(i, params_):
            x = make_x(i)
            zz = params_[-1] + sum(w * xj for w, xj in zip(params_[:-1], x))
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

        params_l1 = fit_logistic([make_x(i) for i in tr],
                                 [data[i][3] for i in tr])

        # train 内 2 折 OOF -> L2（fuse）
        tr_q = sorted({data[i][0].qid for i in tr})
        half = max(1, len(tr_q) // 2)
        folds = (set(tr_q[:half]), set(tr_q[half:]))
        oof = {}
        for ftr, fte in ((folds[0], folds[1]), (folds[1], folds[0])):
            tri = [i for i in tr if data[i][0].qid in ftr]
            tei = [i for i in tr if data[i][0].qid in fte]
            p_ = fit_logistic([make_x(i) for i in tri],
                              [data[i][3] for i in tri])
            for i in tei:
                oof[i] = raw_score(i, p_)

        def l2_x(s, sh):
            return [logit(s), sh, logit(s) * sh]

        l2_fit = [i for i in tr if i in oof]
        l2_raw = [l2_x(oof[i], shares.get(data[i][0].key, 0.5)) for i in l2_fit]
        l2_y = [1.0 if data[i][3] else 0.0 for i in l2_fit]
        d2 = len(l2_raw[0])
        mu2 = [sum(r[j] for r in l2_raw) / len(l2_raw) for j in range(d2)]
        sd2 = [max(1e-6, math.sqrt(sum((r[j] - mu2[j]) ** 2 for r in l2_raw)
                                  / len(l2_raw))) for j in range(d2)]
        params_l2 = fit_logistic(
            [[(r[j] - mu2[j]) / sd2[j] for j in range(d2)] for r in l2_raw], l2_y)

        def fuse_of(i):
            x = l2_x(raw_score(i, params_l1), shares.get(data[i][0].key, 0.5))
            zz = params_l2[-1] + sum(w * xj for w, xj in
                                     zip(params_l2[:-1],
                                         [(xj - m) / s
                                          for xj, m, s in zip(x, mu2, sd2)]))
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

        # test 评测
        k2idx = {data[i][0].key: i for i in range(len(data))}
        te_k = [data[i][0].key for i in te]
        co = [data[i][3] for i in te]
        sc_b4 = [raw_score(i, params_l1) for i in te]
        sc_fuse = [fuse_of(i) for i in te]
        sc_b1 = [ext["b1"].get(k, 0.5) for k in te_k]
        b3_keys = [k for k in te_k if k in ext["b3"]]
        vals = {
            "b1": M.auroc(sc_b1, co), "b4lo": M.auroc(sc_b4, co),
            "fuselo": M.auroc(sc_fuse, co),
        }
        vals["b3"] = (M.auroc([ext["b3"][k] for k in b3_keys],
                              [data[k2idx[k]][3] for k in b3_keys])
                      if b3_keys else float("nan"))
        ece = {"b1": M.ece(sc_b1, co), "b4lo": M.ece(sc_b4, co),
               "fuselo": M.ece(sc_fuse, co)}
        ece["b3"] = (M.ece([ext["b3"][k] for k in b3_keys],
                           [data[k2idx[k]][3] for k in b3_keys])
                     if b3_keys else float("nan"))
        for k in vals:
            agg.setdefault(k, []).append(vals[k])
        for k in ece:
            agg.setdefault("e_" + k, []).append(ece[k])
        print(f"| {db} | {len(te)} | {vals['b1']:.3f} | {vals['b3']:.3f} "
              f"| {vals['b4lo']:.3f} | {vals['fuselo']:.3f} "
              f"| {ece['b1']:.3f} | {ece['b3']:.3f} "
              f"| {ece['b4lo']:.3f} | {ece['fuselo']:.3f} |")

    import statistics as st
    print("\n# macro 平均（11 折） vs 全局 i.i.d.（Phase 2/3，eval 同口径 n≈2,940）")
    glob = {"b1": (0.576, 0.420), "b3": (0.607, 0.412),
            "b4lo": (0.685, 0.138), "fuselo": (0.680, 0.128)}
    names = {"b1": "B1", "b3": "B3", "b4lo": "B4-LODO", "fuselo": "fuse-LODO"}
    for k in ("b1", "b3", "b4lo", "fuselo"):
        a = st.mean(agg[k])
        e = st.mean(agg["e_" + k])
        ga, ge = glob[k]
        print(f"  {names[k]:10s} AUROC {a:.3f} (全局 {ga:.3f}, "
              f"Δ{a - ga:+.3f}) | ECE {e:.3f} (全局 {ge:.3f})")


if __name__ == "__main__":
    main()
