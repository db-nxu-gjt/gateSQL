# -*- coding: utf-8 -*-
"""metrics.py — 置信质量与选择性预测指标（对齐 AAAI 2026 / 选择性预测协议）。

指标：
  - AUROC（区分对错，tie 处理用秩平均）
  - ECE（15 等宽 bin）、Brier、NLL
  - Risk-coverage 曲线与 AURC（越低越好）
  - Selective EX @ 覆盖率 {50%, 70%, 90%}
  - 配对 bootstrap（比较两条 risk-coverage 曲线的 AURC 差）
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def auroc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """correct=True 应得分更高。并列取平均秩（Mann-Whitney U）。"""
    pos = [s for s, c in zip(scores, correct) if c]
    neg = [s for s, c in zip(scores, correct) if not c]
    if not pos or not neg:
        return float("nan")
    allv = sorted(zip(scores, correct))
    ranks: dict[int, float] = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    r_pos = sum(ranks[k] for k, (_, c) in enumerate(allv) if c)
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def ece(scores: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """期望校准误差：把 conf 当作 P(correct)。等宽 bin。"""
    if not scores:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for s, c in zip(scores, correct):
        b = min(n_bins - 1, max(0, int(s * n_bins)))
        bins[b].append((s, c))
    total = len(scores)
    err = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(1 for _, c in b if c) / len(b)
        conf = sum(s for s, _ in b) / len(b)
        err += len(b) / total * abs(acc - conf)
    return err


def brier(scores: Sequence[float], correct: Sequence[bool]) -> float:
    if not scores:
        return float("nan")
    return sum((s - (1.0 if c else 0.0)) ** 2 for s, c in zip(scores, correct)) / len(scores)


def nll(scores: Sequence[float], correct: Sequence[bool], eps: float = 1e-9) -> float:
    if not scores:
        return float("nan")
    tot = 0.0
    for s, c in zip(scores, correct):
        p = min(1.0 - eps, max(eps, s)) if c else min(1.0 - eps, max(eps, 1.0 - s))
        tot += -math.log(p)
    return tot / len(scores)


def risk_coverage(scores: Sequence[float], correct: Sequence[bool]):
    """按置信降序逐点 (coverage, risk)。返回两个等长 list。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n = len(order)
    cov, risk = [], []
    n_corr = 0
    for k, idx in enumerate(order, start=1):
        n_corr += 1 if correct[idx] else 0
        cov.append(k / n)
        risk.append(1.0 - n_corr / k)
    return cov, risk


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """AURC：risk-coverage 曲线梯形积分（越低越好）。"""
    cov, risk = risk_coverage(scores, correct)
    area = 0.0
    for i in range(1, len(cov)):
        area += (cov[i] - cov[i - 1]) * (risk[i] + risk[i - 1]) / 2
    return area


def selective_ex(scores: Sequence[float], correct: Sequence[bool],
                 coverage: float) -> float:
    """固定覆盖率下的 selective accuracy（文档协议：50%/70%/90% 三档）。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    k = max(1, round(coverage * len(order)))
    top = order[:k]
    return sum(1 for i in top if correct[i]) / k


def _aurc_on(scores, correct, idx):
    return aurc([scores[i] for i in idx], [correct[i] for i in idx])


def paired_bootstrap_aurc(scores_a, scores_b, correct, n_boot: int = 10000,
                          seed: int = 0) -> dict:
    """AURC(B) - AURC(A) 的配对 bootstrap（按题目聚类重采样更优，
    这里按候选粒度；delta<0 表示 A 更好）。"""
    rng = random.Random(seed)
    n = len(scores_a)
    deltas = []
    idxs = list(range(n))
    for _ in range(n_boot):
        idx = [idxs[rng.randrange(n)] for _ in range(n)]
        deltas.append(_aurc_on(scores_b, correct, idx)
                      - _aurc_on(scores_a, correct, idx))
    deltas.sort()
    mean = sum(deltas) / n_boot
    lo, hi = deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot) - 1]
    return {"delta_mean": mean, "ci95": [lo, hi],
            "p_delta_lt0": sum(1 for d in deltas if d < 0) / n_boot}


def summarize(scores: Sequence[float], correct: Sequence[bool]) -> dict:
    return {
        "n": len(scores),
        "acc": sum(1 for c in correct if c) / max(1, len(correct)),
        "auroc": round(auroc(scores, correct), 4),
        "ece15": round(ece(scores, correct), 4),
        "brier": round(brier(scores, correct), 4),
        "nll": round(nll(scores, correct), 4),
        "aurc": round(aurc(scores, correct), 4),
        "sel_ex@50": round(selective_ex(scores, correct, 0.50), 4),
        "sel_ex@70": round(selective_ex(scores, correct, 0.70), 4),
        "sel_ex@90": round(selective_ex(scores, correct, 0.90), 4),
    }
