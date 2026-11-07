# -*- coding: utf-8 -*-
"""B3b FTC-Product — token 概率重标定路线（白盒，AAAI 2026 强基线）。

出处: Ramachandran & Sarawagi, "Text-to-SQL Calibration: No Need to Ask –
      Just Rescale Model Probabilities" (arXiv:2411.16742)。
方法: 原始置信 = 生成 SQL 全部 token 概率之积（logprob 求和）；
      再用留出题拟合 Platt 变换 p = sigmoid(a * z + b) 重标定
      （z = log-odds of raw product）。拟合按"题"划分，防止同题候选泄漏。

输入: 带 token_logprobs 的开源模型候选池
输出: confidence jsonl

用法:
  python run_b3_ftc.py --pool PATH --labels PATH --out PATH \
      [--fit-frac 0.3] [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import load_pool, load_labels, write_jsonl  # noqa: E402


def raw_conf(tokens: list[str], lps: list[float]) -> float:
    """FTC 原始置信：全 token 概率积（长度归一可选，默认不归一，
    与 FTC-Product 一致）。"""
    return math.exp(sum(lps))


def logit(p: float, eps: float = 1e-9) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def fit_platt(z: list[float], y: list[bool], iters: int = 100) -> tuple[float, float]:
    """最小化 NLL 的 Platt scaling（简单梯度下降，无需 sklearn）。"""
    a, b = 1.0, 0.0
    n = len(z)
    lr = 0.1
    for _ in range(iters):
        ga = gb = 0.0
        for zi, yi in zip(z, y):
            p = 1.0 / (1.0 + math.exp(-(a * zi + b)))
            d = p - (1.0 if yi else 0.0)
            ga += d * zi
            gb += d
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-frac", type=float, default=0.3,
                    help="用于拟合 Platt 的题目比例（按题划分，不按候选）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool(args.pool, require_logprobs=True)
    labels = load_labels(args.labels)
    data = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue
        z = logit(raw_conf(c.tokens, c.token_logprobs))
        data.append((c, z, lab["correct"]))

    # 按题划分 fit / eval（同题候选不得跨集，防泄漏）
    qids = sorted({c.qid for c, _, _ in data})
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    n_fit = max(1, int(len(qids) * args.fit_frac))
    fit_qids = set(qids[:n_fit])
    fit = [(z, y) for c, z, y in data if c.qid in fit_qids]
    eva = [(c, z, y) for c, z, y in data if c.qid not in fit_qids]

    a, b = fit_platt([z for z, _ in fit], [y for _, y in fit])
    print(f"[ftc] Platt a={a:.4f} b={b:.4f} (fit={len(fit)}, eval={len(eva)})")

    rows = []
    for c, z, _ in eva:
        p = 1.0 / (1.0 + math.exp(-(a * z + b)))
        rows.append({"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                     "conf": p, "method": "b3_ftc_product"})
    write_jsonl(args.out, rows)
    print(f"[done] {len(rows)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
