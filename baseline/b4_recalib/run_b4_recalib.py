# -*- coding: utf-8 -*-
"""B4 Recalibration-only — 生成模型概率重标定（EMNLP 2025 路线）。

出处: "Calibrating LLMs for Text-to-SQL Parsing by Leveraging Sub-clause
      Frequencies", EMNLP 2025 (arXiv:2505.23804) —— 用子句频率特征对
      生成概率做后验重标定。本实现：
        base conf = FTC 原始概率（全 token 概率积，需开源池 logprobs），
        特征 = SQL 子句频率（SELECT 列数 / JOIN 数 / GROUP BY / 子查询等），
        重标定 = 逻辑回归 p = sigmoid(w·x + b)（按题划分拟合，防泄漏）。

输入: 带 token_logprobs 的开源模型候选池 + 标签
输出: confidence jsonl

用法:
  python run_b4_recalib.py --pool PATH --labels PATH --out PATH \
      [--fit-frac 0.3] [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import load_pool, load_labels, write_jsonl  # noqa: E402

CLAUSE_RE = {
    "n_select_cols": r"(?<=select)[^;]*?(?= from)",
    "n_joins": r"\bjoin\b",
    "n_subqueries": r"\(\s*select\b",
    "n_aggregates": r"\b(count|sum|avg|min|max)\s*\(",
    "n_where_preds": r"(?<=where).*?(?= group by| order by| limit|;|$)",
}


def clause_features(sql: str) -> list[float]:
    s = " ".join(sql.lower().split())
    m_sel = re.search(CLAUSE_RE["n_select_cols"], s)
    n_sel = 0
    if m_sel:
        n_sel = m_sel.group(0).count(",") + 1
    n_join = len(re.findall(CLAUSE_RE["n_joins"], s))
    n_sub = len(re.findall(CLAUSE_RE["n_subqueries"], s))
    n_agg = len(re.findall(CLAUSE_RE["n_aggregates"], s))
    has_group = 1.0 if "group by" in s else 0.0
    n_group_cols = 0.0
    mg = re.search(r"group by ([^;]+?)(?= order by| limit| having|;|$)", s)
    if mg:
        n_group_cols = mg.group(1).count(",") + 1
    has_order = 1.0 if "order by" in s else 0.0
    has_limit = 1.0 if re.search(r"\blimit\b", s) else 0.0
    has_where = 1.0 if " where " in f" {s} " else 0.0
    n_where = len(re.findall(r"\band\b|\bor\b", s)) + (1.0 if has_where else 0.0)
    has_having = 1.0 if "having" in s else 0.0
    n_setop = len(re.findall(r"\bunion all\b|\bunion\b|\bintersect\b|\bexcept\b", s))
    sql_len = min(len(s) / 200.0, 5.0)
    return [n_sel, n_join, n_sub, n_agg, has_group, n_group_cols, has_order,
            has_limit, has_where, n_where, has_having, n_setop, sql_len]


FEAT_DIM = 13


def fit_logistic(X: list[list[float]], y: list[bool], iters: int = 300,
                 lr: float = 0.5) -> list[float]:
    """带 L2 的逻辑回归（纯 python 梯度下降），特征已标准化。"""
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            g = p - (1.0 if yi else 0.0)
            for j in range(d):
                gw[j] += g * xi[j]
            gb += g
        for j in range(d):
            w[j] -= lr * (gw[j] / n + 1e-4 * w[j])
        b -= lr * gb / n
    return w + [b]


def isotonic_fit(xs: list[float], ys: list[float]):
    """PAVA 拟合单调不减映射 f: x -> y（块均值，含并列平台）。返回查询函数。"""
    import bisect as bi
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    blocks = []  # [x_start, x_end, mean, weight]
    for i in order:
        blocks.append([xs[i], xs[i], ys[i], 1.0])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2] + 1e-12:
            b2 = blocks.pop()
            b1 = blocks.pop()
            w = b1[3] + b2[3]
            m = (b1[2] * b1[3] + b2[2] * b2[3]) / w
            blocks.append([b1[0], b2[1], m, w])
    starts = [b[0] for b in blocks]

    def f(x: float) -> float:
        j = bi.bisect_right(starts, x) - 1
        return blocks[j][2] if j >= 0 else blocks[0][2]
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-frac", type=float, default=0.3)
    ap.add_argument("--isotonic", action="store_true",
                    help="fit 集 2 折 OOF 拟合 isotonic 后校准（修高端过自信；"
                         "isotonic 平台化同时改变分数并列结构，供 conformal 用）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool, require_logprobs=True)
    labels = load_labels(args.labels)
    data = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue
        base = math.exp(sum(c.token_logprobs))  # FTC 原始概率
        z = math.log(max(base, 1e-12))          # log 空间更稳
        data.append((c, z, clause_features(c.sql), lab["correct"]))

    # 标准化特征（fit 集统计量）
    qids = sorted({c.qid for c, _, _, _ in data})
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    n_fit = max(1, int(len(qids) * args.fit_frac))
    fit_qids = set(qids[:n_fit])

    fit_idx = [i for i, d in enumerate(data) if d[0].qid in fit_qids]
    eva_idx = [i for i, d in enumerate(data) if d[0].qid not in fit_qids]
    if not fit_idx or not eva_idx:
        raise SystemExit("fit/eval 划分为空，检查 --fit-frac")

    dim = len(data[0][2])
    mu = [sum(data[i][2][j] for i in fit_idx) / len(fit_idx) for j in range(dim)]
    sd = [max(1e-6, math.sqrt(sum((data[i][2][j] - mu[j]) ** 2
                                  for i in fit_idx) / len(fit_idx)))
          for j in range(dim)]

    def norm(feat):
        return [(fj - m) / s for fj, m, s in zip(feat, mu, sd)]

    def make_x(i):
        return norm(data[i][2]) + [data[i][1] / 10.0]  # +logprob 特征

    def raw_score(i, params_):
        x = make_x(i)
        zz = params_[-1] + sum(w * xj for w, xj in zip(params_[:-1], x))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

    iso_fn = None
    if args.isotonic:
        # fit 集内 2 折 OOF 分数拟合 isotonic（out-of-fold，无泄漏）。
        fq = sorted(fit_qids)
        rng2 = random.Random(args.seed + 1)
        rng2.shuffle(fq)
        half = max(1, len(fq) // 2)
        folds = (set(fq[:half]), set(fq[half:]))
        oof = []
        for tr, te in ((folds[0], folds[1]), (folds[1], folds[0])):
            tr_idx = [i for i in fit_idx if data[i][0].qid in tr]
            te_idx = [i for i in fit_idx if data[i][0].qid in te]
            p_ = fit_logistic([make_x(i) for i in tr_idx],
                              [data[i][3] for i in tr_idx])
            oof += [(raw_score(i, p_), 1.0 if data[i][3] else 0.0)
                    for i in te_idx]
        iso_fn = isotonic_fit([s for s, _ in oof], [y for _, y in oof])

    params = fit_logistic([make_x(i) for i in fit_idx],
                          [data[i][3] for i in fit_idx])

    method = "b4_subclause_recalib_iso" if args.isotonic else "b4_subclause_recalib"
    rows = []
    for i in eva_idx:
        c, z, feat, _ = data[i]
        p = raw_score(i, params)
        if iso_fn is not None:
            p = iso_fn(p)
        rows.append({"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                     "conf": p, "method": method})
    write_jsonl(args.out, rows)
    print(f"[done] fit={len(fit_idx)} eval={len(eva_idx)} -> {args.out}")


if __name__ == "__main__":
    main()
