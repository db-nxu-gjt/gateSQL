# -*- coding: utf-8 -*-
"""B8 Conformal 兜底 — 对任意置信方法加 conformal 准入控制。

出处: Angelopoulos & Bates, "Conformal Prediction: A Gentle Introduction" (2021);
      Sadinle et al. (2019) 最少歧义集合分类（conformal hypothesis testing）。
方法: 把"该候选是错的"当作零假设做 conformal 检验。
      非一致性分数 s = 1 - conf（越大越可疑）。
      p 值 p = (1 + #{校准错误样本 s_i ≥ s}) / (n_wrong + 1)，
      仅当 p ≤ α 才允许自动执行。
      有限样本保证（可交换性下）: P(接受 | 候选错误) ≤ α，
      即错误候选漏放进自动执行区的比例 ≤ α；
      接受集的经验错误率另以 Clopper-Pearson 区间审计报告。

输入: 任一方法的 confidence jsonl + 标签 jsonl（如 B7 裁判输出）+ 候选池
输出: 接受决策 jsonl + 报告 json

用法:
  python run_b8_conformal.py --conf PATH --pool PATH --labels PATH --out PATH \
      [--alpha 0.02] [--cal-frac 0.4] [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import load_pool, load_labels, write_jsonl  # noqa: E402


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """二项精确置信区间（用于覆盖率检验报告）。"""
    def beta_inv(a: float, b: float, x: float) -> float:
        # 二分求 Beta 分布分位数（n 通常不大，足够精确）
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            # 正则化不完全贝塔（数值积分替代，k/n 规模下稳定）
            if _beta_cdf(mid, a, b) < x:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _beta_cdf(x: float, a: float, b: float) -> float:
        # 通过网格积分近似（精度 1e-4，报告用足够）
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        n_steps = 2000
        dx = x / n_steps
        total = 0.0
        for i in range(1, n_steps + 1):
            t = (i - 0.5) * dx
            total += t ** (a - 1) * (1 - t) ** (b - 1) * dx
        # 归一化常数
        log_norm = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
        return total * math.exp(log_norm)

    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else beta_inv(k, n - k + 1, 0.025)
    hi = 1.0 if k == n else beta_inv(k + 1, n - k, 0.975)
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="待兜底的置信文件（如 B7）")
    ap.add_argument("--pool", required=True, help="候选池（取 question 信息）")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="接受决策输出 jsonl")
    ap.add_argument("--report", default=None, help="覆盖率报告 json")
    ap.add_argument("--alpha", type=float, default=0.02,
                    help="目标：自动执行错误率 ≤ alpha（文档目标 2%）")
    ap.add_argument("--cal-frac", type=float, default=0.4,
                    help="校准题目比例（按题划分）")
    ap.add_argument("--stratify", default="none",
                    choices=["none", "difficulty", "score"],
                    help="Mondrian 分层校准：difficulty=按题难度层内校准；"
                         "score=按置信分桶层内校准；none=全局（原版行为）")
    ap.add_argument("--n-buckets", type=int, default=3,
                    help="stratify=score 时的等宽分桶数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = {c.key: c for c in load_pool(args.pool)}
    labels = load_labels(args.labels)

    items = []
    for r in open(args.conf, encoding="utf-8"):
        if not r.strip():
            continue
        d = __import__("json").loads(r)
        k = (d["qid"], d["source"], d["cand_idx"])
        lab = labels.get(k)
        if lab is None or k not in pool:
            continue
        items.append((k, float(d["conf"]), lab["correct"]))

    # 按题划分 cal / test
    qids = sorted({k[0] for k, _, _ in items})
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    n_cal = max(1, int(len(qids) * args.cal_frac))
    cal_qids = set(qids[:n_cal])
    cal = [(k, s, ok) for k, s, ok in items if k[0] in cal_qids]
    test = [(k, s, ok) for k, s, ok in items if k[0] not in cal_qids]

    # Mondrian 分层：层内独立做 conformal 检验（层保证 => 混合保证仍成立）。
    # stratify=none 时层键恒 0，行为与全局版完全一致。
    from collections import defaultdict

    if args.stratify == "difficulty":
        def skey(k, s):
            return (pool.get(k).difficulty if pool.get(k) is not None
                    and pool.get(k).difficulty else "unknown")
    elif args.stratify == "score":
        def skey(k, s):
            return min(args.n_buckets - 1, int(s * args.n_buckets))
    else:
        def skey(k, s):
            return 0

    cal_l: dict = defaultdict(list)
    test_l: dict = defaultdict(list)
    for k, s, ok in cal:
        cal_l[skey(k, s)].append((k, s, ok))
    for k, s, ok in test:
        test_l[skey(k, s)].append((k, s, ok))

    # conformal 检验（H0: 候选错误），逐层:
    #   p = (1 + #{层内校准错误分数 s_i ≤ s}) / (n_wrong_layer + 1)
    #   仅当 p ≤ α 接受；层内无校准错误样本时无法校准 -> 全拒（保守）。
    import bisect

    accept = []
    per_layer = {}
    for lay in sorted(test_l, key=str):
        titems = test_l[lay]
        cw = sorted((1.0 - s) for _, s, ok in cal_l.get(lay, []) if not ok)
        acc_here = []
        if cw:
            for k, s, ok in titems:
                le = bisect.bisect_right(cw, 1.0 - s)
                if (1 + le) / (len(cw) + 1) <= args.alpha:
                    acc_here.append((k, s))
        accept += acc_here
        lay_wrong = sum(1 for _, _, ok in titems if not ok)
        lay_acc_wrong = sum(1 for k, _ in acc_here if not labels[k]["correct"])
        per_layer[str(lay)] = {
            "n_test": len(titems), "n_cal_wrong": len(cw),
            "n_accept": len(acc_here), "accepted_wrong": lay_acc_wrong,
            "leak": round(lay_acc_wrong / max(1, lay_wrong), 4),
            "coverage": round(len(acc_here) / max(1, len(titems)), 4),
        }

    threshold = min((s for _, s in accept), default=1.0)

    acc_wrong = sum(1 for k, _ in accept if not labels[k]["correct"])
    test_wrong = sum(1 for _, _, ok in test if not ok)
    err = acc_wrong / max(1, len(accept))                      # 接受集错误率（审计）
    leak = acc_wrong / max(1, test_wrong)                      # 错误样本漏放率（被保证量）
    cov = len(accept) / max(1, len(test))
    lo, hi = clopper_pearson(acc_wrong, max(1, len(accept)))

    rows = [{"qid": k[0], "source": k[1], "cand_idx": k[2],
             "accept": True, "conf": s, "method": "b8_conformal"}
            for k, s in accept]
    write_jsonl(args.out, rows)

    report = {
        "alpha_target": args.alpha,
        "stratify": args.stratify,
        "n_cal_wrong": sum(v["n_cal_wrong"] for v in per_layer.values()),
        "n_test": len(test), "n_accept": len(accept),
        "guaranteed_leak_rate": ("P(accept|wrong) <= %.3f" % args.alpha),
        "empirical_leak_rate": round(leak, 4),
        "leak_guarantee_holds": leak <= args.alpha + 1e-9,
        "accepted_set_error_rate": round(err, 4),
        "error_ci95": [round(lo, 4), round(hi, 4)],
        "coverage": round(cov, 4),
        "per_layer": per_layer,
    }
    rp = args.report or os.path.splitext(args.out)[0] + "_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        __import__("json").dump(report, f, indent=2, ensure_ascii=False)
    print("[b8]", __import__("json").dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
