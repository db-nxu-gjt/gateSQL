# -*- coding: utf-8 -*-
"""revision_round1.py — 第一轮修订（R1-1/R1-4/R1-5 + Mondrian 层内样本量）全部重算。

R1-1  多 seed 主表重跑: seeds 0-4 重拟合 B4/fuse（同 run_b4_recalib/run_fuse
      协议: 按题 30% fit / 70% eval, exec_fail 保留计 wrong），报 mean±std；
      并在 seed-0 Protocol A 上做题目级聚类 bootstrap 显著性检验
      （Fuse-prod vs B4 / Fuse-prod vs B1 / B4 vs B1，ΔAUROC 与 ΔselEX@30）。
R1-4  B4 消融: z-only / clause-only / full（seed-0）+ 特征重要性 |w|·sd 排序。
R1-5  exec_fail 口径敏感性: 计 wrong（现行）vs 排除（correct+silent），B0/B1/B4/fuse_prod。
R1-2  Mondrian 层内校准样本量: 重算 B8 叠 B4（α=2%, cal 40% 按题, seeds 0-4,
      global/difficulty/score-3桶），逐层 n_cal_wrong / n_test_wrong，
      并核对保证成立 seed 数与论文 Table 6（2/5, 4/5, 4/5）一致。

输出: results/revision_round1.md + paper/results/main_table.xlsx 追加 5 个 sheet。
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import statistics as st
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "tools"))
from common import metrics as M  # noqa: E402
from common.schema import load_labels, load_pool  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402
from gen_excel import write_sheet, load_conf  # noqa: E402

POOLDIR = os.path.join(_ROOT, "data", "pools")
RESDIR = os.path.join(_ROOT, "results")
XLDIR = os.path.join(_ROOT, "results")
SEEDS = [0, 1, 2, 3, 4]


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


# ---------------- 数据加载 ----------------
pool = load_pool(os.path.join(POOLDIR, "bird300_ds_pool.jsonl"),
                 require_logprobs=True)
labels = load_labels(os.path.join(POOLDIR, "bird300_ds_labels.jsonl"))
data = []
for c in pool:
    lab = labels.get(c.key)
    if lab is None:
        continue
    base = math.exp(sum(c.token_logprobs))
    z = math.log(max(base, 1e-12))  # 与 run_b4_recalib 完全一致（含截断）
    data.append((c, z, clause_features(c.sql), lab["correct"]))
print(f"[data] candidates={len(data)} (with logprobs & labels)", flush=True)

shares = {}
for fn in ("bird300_ds_b2_nuc10.jsonl", "bird300_ds_b2_beam10.jsonl"):
    shares.update(load_conf(fn))
print(f"[data] shares={len(shares)}", flush=True)

b1 = load_conf("bird300_ds_b1.jsonl")
b3 = load_conf("bird300_ds_b3_sac.jsonl")
b4_s0_file = load_conf("bird300_ds_b4.jsonl")          # 现行 seed-0 结果
fp_s0_file = load_conf("bird300_ds_fuse_prod.jsonl")
fs_s0_file = load_conf("bird300_ds_fuse_stack.jsonl")

qid_pool = sorted({d[0].qid for d in data})
N_FIT = max(1, int(len(qid_pool) * 0.3))


def split(seed: int):
    q = qid_pool[:]
    random.Random(seed).shuffle(q)
    return set(q[:N_FIT])


def eval_keys(fit_qids):
    ks = sorted(d[0].key for d in data if d[0].qid not in fit_qids)
    return ks


# ---------------- B4 / fuse 拟合（复现原管线） ----------------
def fit_b4(fit_idx, mode: str):
    """mode: full(13+1) / z(1) / clause(13)。返回 (score_fn, importance list)。"""
    if mode == "full":
        dim = 13
    elif mode == "clause":
        dim = 13
    else:
        dim = 0
    fit_sds = []
    if mode in ("full", "clause"):
        mu = [sum(data[i][2][j] for i in fit_idx) / len(fit_idx)
              for j in range(dim)]
        sd = [max(1e-6, math.sqrt(sum((data[i][2][j] - mu[j]) ** 2
                                      for i in fit_idx) / len(fit_idx)))
              for j in range(dim)]
        fit_sds = sd
    else:
        mu, sd = [], []

    def x_of(i):
        if mode == "full":
            return ([(f - m) / s for f, m, s in zip(data[i][2], mu, sd)]
                    + [data[i][1] / 10.0])
        if mode == "clause":
            return [(f - m) / s for f, m, s in zip(data[i][2], mu, sd)]
        return [data[i][1] / 10.0]

    w = fit_logistic([x_of(i) for i in fit_idx],
                     [data[i][3] for i in fit_idx])

    def score(i):
        return sig(w[-1] + sum(wj * xj for wj, xj in zip(w[:-1], x_of(i))))

    imp = None
    if mode == "full":
        names = ["n_select_cols", "n_joins", "n_subqueries", "n_aggregates",
                 "has_group", "n_group_cols", "has_order", "has_limit",
                 "has_where", "n_where_preds", "has_having", "n_setop",
                 "sql_len", "+logprob(z/10)"]
        zsd = math.sqrt(sum((data[i][1] - sum(d[1] for d in data) / len(data)) ** 2
                            for i in fit_idx) / len(fit_idx))
        sds = fit_sds + [max(1e-6, zsd / 10.0)]
        imp = sorted((abs(wj) * s_, nm) for wj, s_, nm in zip(w[:-1], sds, names))
        imp.reverse()
    return score, x_of, imp


def fuse_stack_scores(seed, fit_qids, fit_idx, x_of, raw_of):
    """复现 run_fuse.py L2: [logit(OOF b4), share, 交互] 逻辑回归。
    raw_of = 全拟合 B4 分数（eval 候选的 L2 输入）；OOF 折内使用与全模型
    相同的 fit 集标准化（忠实复现原实现）。"""
    fq = sorted(fit_qids)
    random.Random(seed + 1).shuffle(fq)
    half = max(1, len(fq) // 2)
    folds = (set(fq[:half]), set(fq[half:]))
    oof = {}
    for tr, te in ((folds[0], folds[1]), (folds[1], folds[0])):
        tri = [i for i in fit_idx if data[i][0].qid in tr]
        tei = [i for i in fit_idx if data[i][0].qid in te]
        p_ = fit_logistic([x_of(i) for i in tri], [data[i][3] for i in tri])
        for i in tei:
            oof[i] = sig(p_[-1] + sum(wj * xj for wj, xj in
                                      zip(p_[:-1], x_of(i))))
    l2f = [i for i in fit_idx if i in oof]
    rows2 = []
    for i in l2f:
        lg = logit(oof[i])
        sh = shares.get(data[i][0].key, 0.5)
        rows2.append([lg, sh, lg * sh])
    y2 = [1.0 if data[i][3] else 0.0 for i in l2f]
    d2 = 3
    mu2 = [sum(r[j] for r in rows2) / len(rows2) for j in range(d2)]
    sd2 = [max(1e-6, math.sqrt(sum((r[j] - mu2[j]) ** 2 for r in rows2)
                             / len(rows2))) for j in range(d2)]
    w2 = fit_logistic([[(r[j] - mu2[j]) / sd2[j] for j in range(d2)]
                       for r in rows2], y2)

    def fuse(i):
        lg = logit(raw_of(i))
        sh = shares.get(data[i][0].key, 0.5)
        x = [lg, sh, lg * sh]
        return sig(w2[-1] + sum(wj * ((xj - m) / s_)
                                for wj, xj, m, s_ in zip(w2[:-1], x, mu2, sd2)))
    return fuse


# ---------------- 指标 ----------------
def sel_ex(cf: dict, keys, cov: float) -> float:
    ranked = sorted(keys, key=lambda k: (-cf[k], k))
    take = ranked[:max(1, int(len(ranked) * cov))]
    return sum(labels[k]["correct"] for k in take) / len(take)


def eval_metrics(cf: dict, keys, count_exec_fail: bool = True):
    ks = keys if count_exec_fail else [k for k in keys
                                       if labels[k]["raw"] != "exec_fail"]
    co = [labels[k]["correct"] for k in ks]
    sc = [cf[k] for k in ks]
    return {"auroc": M.auroc(sc, co), "ece": M.ece(sc, co),
            "s50": sel_ex(cf, ks, 0.5), "s30": sel_ex(cf, ks, 0.3),
            "n": len(ks), "ex": sum(co) / len(ks)}


# ---------------- R1-1: 多 seed ----------------
multi = defaultdict(list)          # method -> list of metric dicts
seed0 = {}                         # method -> (cf dict on seed0 eval keys)
for seed in SEEDS:
    fit_qids = split(seed)
    fit_idx = [i for i, d in enumerate(data) if d[0].qid in fit_qids]
    keys = eval_keys(fit_qids)

    raw4, x4, _ = fit_b4(fit_idx, "full")
    fuse_s = fuse_stack_scores(seed, fit_qids, fit_idx, x4, raw4)

    cf_b4, cf_fmin, cf_fprod, cf_fstack = {}, {}, {}, {}
    for i, d in enumerate(data):
        if d[0].qid in fit_qids:
            continue
        k = d[0].key
        r = raw4(i)
        sh = shares.get(k, 0.5)
        cf_b4[k] = r
        cf_fmin[k] = min(r, sh)
        cf_fprod[k] = r * sh
        cf_fstack[k] = fuse_s(i)
    cf_b1 = {k: b1[k] for k in keys}
    cf_b3 = {k: b3[k] for k in keys}
    ex = sum(labels[k]["correct"] for k in keys) / len(keys)
    cf_b0 = {k: 0.5 for k in keys}          # B0 always-execute（并列分）
    confs = {"B0 Always-Execute": cf_b0, "B1 Verbalized": cf_b1,
             "B3 SAC-Avg": cf_b3, "B4 Clause Recalib": cf_b4,
             "Fuse-min": cf_fmin, "Fuse-prod": cf_fprod,
             "Fuse-stack (L2)": cf_fstack}
    for nm, cf in confs.items():
        m = eval_metrics(cf, keys)
        if nm == "B0 Always-Execute":
            m["s50"] = m["s30"] = ex        # 与 gen_excel 口径一致
        multi[nm].append(m)
    if seed == 0:
        seed0 = confs
        # 校验: 重算 vs 现行 jsonl
        d_b4 = max(abs(cf_b4[k] - b4_s0_file[k]) for k in keys)
        d_fp = max(abs(cf_fprod[k] - fp_s0_file[k]) for k in keys)
        d_fs = max(abs(cf_fstack[k] - fs_s0_file[k]) for k in keys)
        print(f"[check] seed0 max|d| b4={d_b4:.2e} fuse_prod={d_fp:.2e} "
              f"fuse_stack={d_fs:.2e}", flush=True)
    print(f"[seed {seed}] done "
          f"(B4 {multi['B4 Clause Recalib'][-1]['auroc']:.4f}, "
          f"FP {multi['Fuse-prod'][-1]['auroc']:.4f})", flush=True)

# ---------------- R1-1: 显著性（题目级聚类 bootstrap, seed-0） ----------------
keys0 = sorted(seed0["B4 Clause Recalib"])
q2keys = defaultdict(list)
for k in keys0:
    q2keys[k[0]].append(k)
qids0 = sorted(q2keys)


def pair_boot(m_a: dict, m_b: dict, n_boot=2000, seed=123):
    """delta = metric(A) - metric(B)，按题目聚类重采样。"""
    rng = random.Random(seed)
    da, ds = [], []
    for _ in range(n_boot):
        samp = []
        for _ in range(len(qids0)):
            samp += q2keys[rng.choice(qids0)]
        ca = [labels[k]["correct"] for k in samp]
        da.append(M.auroc([m_a[k] for k in samp], ca)
                  - M.auroc([m_b[k] for k in samp], ca))
        ds.append(sel_ex(m_a, samp, 0.3) - sel_ex(m_b, samp, 0.3))
    def pack(v):
        v = sorted(v)
        return {"delta": round(st.mean(v), 4),
                "ci95": [round(v[int(0.025 * n_boot)], 4),
                         round(v[int(0.975 * n_boot) - 1], 4)],
                "p_le0": round(sum(1 for x in v if x <= 0) / n_boot, 4)}
    return pack(da), pack(ds)


FP = seed0["Fuse-prod"]
pairs = {
    "Fuse-prod vs B4": pair_boot(FP, seed0["B4 Clause Recalib"]),
    "Fuse-prod vs B1": pair_boot(FP, seed0["B1 Verbalized"]),
    "B4 vs B1": pair_boot(seed0["B4 Clause Recalib"],
                          seed0["B1 Verbalized"]),
}
print("[boot] significance done", flush=True)

# ---------------- R1-4: B4 消融（seed-0） ----------------
fit_qids0 = split(0)
fit_idx0 = [i for i, d in enumerate(data) if d[0].qid in fit_qids0]
keys0b = eval_keys(fit_qids0)
abl = {}
for mode, nm in (("z", "z-only (+logprob)"),
                 ("clause", "clause-only (13 feats)"),
                 ("full", "full (13 feats + z)")):
    sc_fn, _, imp_ = fit_b4(fit_idx0, mode)
    cf = {data[i][0].key: sc_fn(i) for i in range(len(data))
          if data[i][0].qid not in fit_qids0}
    abl[nm] = eval_metrics(cf, keys0b)
importance = imp_  # full 模型 |w|*sd 排序
print("[abl] done", flush=True)

# ---------------- R1-5: exec_fail 口径敏感性（seed-0） ----------------
sens = {}
for nm in ("B0 Always-Execute", "B1 Verbalized", "B4 Clause Recalib",
           "Fuse-prod"):
    cf = dict(seed0[nm])
    a = eval_metrics(cf, keys0b, count_exec_fail=True)
    b = eval_metrics(cf, keys0b, count_exec_fail=False)
    if nm == "B0 Always-Execute":   # always-execute：selEX = 同集合 EX
        a["s50"] = a["s30"] = a["ex"]
        b["s50"] = b["s30"] = b["ex"]
    sens[nm] = (a, b)
print("[sens] done", flush=True)

# ---------------- R1-2: Mondrian 层内样本量 + 核对 ----------------
diff_of = {d[0].key: (d[0].difficulty or "unknown") for d in data}
mond = {}
for strat in ("global", "difficulty", "score"):
    holds = 0
    layer_rows = []
    for seed in SEEDS:
        cf = b4_s0_file
        keys = sorted(set(cf) & set(labels))
        qids = sorted({k[0] for k in keys})
        rq = qids[:]
        random.Random(seed).shuffle(rq)
        cal_q = set(rq[:max(1, int(len(rq) * 0.4))])

        def skey(k, s):
            if strat == "difficulty":
                return diff_of.get(k, "unknown")
            if strat == "score":
                return min(2, int(s * 3))
            return 0

        cal_l, test_l = defaultdict(list), defaultdict(list)
        for k in keys:
            s = cf[k]
            if k[0] in cal_q:
                cal_l[skey(k, s)].append((k, s))
            else:
                test_l[skey(k, s)].append((k, s))
        acc_wrong = 0
        test_wrong = 0
        accept = 0
        for lay, items in cal_l.items():
            cw = sorted(1.0 - s for k, s in items if not labels[k]["correct"])
            n_c = len(cw)
            for k, s in test_l.get(lay, []):
                bad = not labels[k]["correct"]
                test_wrong += bad
                if cw:
                    le = bisect.bisect_right(cw, 1.0 - s)
                    if (1 + le) / (n_c + 1) <= 0.02:
                        accept += 1
                        acc_wrong += bad
        leak = acc_wrong / max(1, test_wrong)
        ok = leak <= 0.02 + 1e-9
        holds += int(ok)
        for lay in sorted(set(cal_l) | set(test_l), key=str):
            cw_n = sum(1 for k, _ in cal_l.get(lay, [])
                       if not labels[k]["correct"])
            tw_n = sum(1 for k, _ in test_l.get(lay, [])
                       if not labels[k]["correct"])
            layer_rows.append([strat, seed, str(lay), len(cal_l.get(lay, [])),
                               cw_n, len(test_l.get(lay, [])), tw_n])
    mond[strat] = {"holds": holds, "layers": layer_rows}
print(f"[mond] holds global={mond['global']['holds']}/5 "
      f"difficulty={mond['difficulty']['holds']}/5 "
      f"score={mond['score']['holds']}/5", flush=True)

# ---------------- 输出 markdown ----------------
os.makedirs(RESDIR, exist_ok=True)
L = []
L.append("# Revision Round 1 — 重算结果（R1-1/R1-2/R1-4/R1-5）\n")
L.append("协议与原实验完全一致：按题 30% fit / 70% eval，exec_fail 保留计 wrong，"
         "确定性平局处理；Protocol A（白盒交集）。\n")

L.append("## R1-1 多 seed 主表（seeds 0-4, eval n=4,200/seed）\n")
L.append("| Method | AUROC | ECE15 | selEX@50 | selEX@30 |")
L.append("|---|---|---|---|---|")
order = ["B0 Always-Execute", "B1 Verbalized", "B3 SAC-Avg",
         "B4 Clause Recalib", "Fuse-min", "Fuse-prod", "Fuse-stack (L2)"]


def ms_str(vals, k):
    xs = [x[k] for x in vals]
    return f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}"


for nm in order:
    row = [nm] + [ms_str(multi[nm], k) for k in ("auroc", "ece", "s50", "s30")]
    L.append("| " + " | ".join(row) + " |")

L.append("\n各 seed 明细：\n")
L.append("| Method | seed | AUROC | ECE15 | selEX@50 | selEX@30 |")
L.append("|---|---|---|---|---|---|")
for nm in order:
    for si, m in enumerate(multi[nm]):
        L.append(f"| {nm} | {si} | {m['auroc']:.4f} | {m['ece']:.4f} | "
                 f"{m['s50']:.4f} | {m['s30']:.4f} |")

L.append("\n### 显著性（题目级聚类 bootstrap, 2,000 次, seed-0 Protocol A）\n")
L.append("| 比较 | ΔAUROC [95% CI] | p(Δ≤0) | ΔselEX@30 [95% CI] | p(Δ≤0) |")
L.append("|---|---|---|---|---|")
for nm, (a, s) in pairs.items():
    L.append(f"| {nm} | {a['delta']:+.4f} [{a['ci95'][0]:+.4f}, "
             f"{a['ci95'][1]:+.4f}] | {a['p_le0']} | "
             f"{s['delta']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | "
             f"{s['p_le0']} |")

L.append("\n## R1-4 B4 消融（seed-0, eval n=4,200）\n")
L.append("| 配置 | AUROC | ECE15 | selEX@50 | selEX@30 |")
L.append("|---|---|---|---|---|")
for nm, m in abl.items():
    L.append(f"| {nm} | {m['auroc']:.4f} | {m['ece']:.4f} | "
             f"{m['s50']:.4f} | {m['s30']:.4f} |")
L.append("\n特征重要性（full 模型 |标准化系数|×fit 集特征 sd，降序）：\n")
for v, nm in importance:
    L.append(f"- {nm}: {v:.3f}")

L.append("\n## R1-5 exec_fail 口径敏感性（seed-0）\n")
L.append("| Method | 口径 | n | AUROC | ECE15 | selEX@50 | selEX@30 |")
L.append("|---|---|---|---|---|---|---|")
for nm, (a, b) in sens.items():
    L.append(f"| {nm} | 计 wrong（主表口径） | {a['n']} | {a['auroc']:.4f} | "
             f"{a['ece']:.4f} | {a['s50']:.4f} | {a['s30']:.4f} |")
    L.append(f"| {nm} | 排除 exec_fail | {b['n']} | {b['auroc']:.4f} | "
             f"{b['ece']:.4f} | {b['s50']:.4f} | {b['s30']:.4f} |")

L.append("## R1-2 Mondrian 层内校准样本量（B8 叠 B4, α=2%, cal 40%, seeds 0-4）\n")
L.append(f"- 保证成立/超标：global {mond['global']['holds']}/5 成立"
         f"（超标 {5 - mond['global']['holds']}/5，论文 Over 2/5）、"
         f"difficulty {mond['difficulty']['holds']}/5 成立"
         f"（超标 {5 - mond['difficulty']['holds']}/5，论文 4/5）、"
         f"score {mond['score']['holds']}/5 成立"
         f"（超标 {5 - mond['score']['holds']}/5，论文 4/5）\n")
L.append("| strat | seed | layer | n_cal | n_cal_wrong | n_test | n_test_wrong |")
L.append("|---|---|---|---|---|---|---|")
for strat in ("global", "difficulty", "score"):
    for r in mond[strat]["layers"]:
        L.append("| " + " | ".join(str(x) for x in r) + " |")

md = "\n".join(L) + "\n"
with open(os.path.join(RESDIR, "revision_round1.md"), "w",
          encoding="utf-8") as f:
    f.write(md)
print(f"[out] {os.path.join(RESDIR, 'revision_round1.md')}", flush=True)

# ---------------- xlsx 追加 sheet ----------------
from openpyxl import load_workbook  # noqa: E402
xlp = os.path.join(XLDIR, "main_table.xlsx")
wb = load_workbook(xlp)
for s in ("MultiSeed", "Significance", "Ablation", "ExecFailSens",
          "MondrianLayers"):
    if s in wb.sheetnames:
        del wb[s]

write_sheet(wb, "MultiSeed",
            ["Method", "AUROC (mean±std)", "ECE15 (mean±std)",
             "selEX@50 (mean±std)", "selEX@30 (mean±std)"],
            [[nm] + [ms_str(multi[nm], k) for k in ("auroc", "ece", "s50", "s30")]
             for nm in order],
            note="5 refit seeds (0-4), 30% fit by question per seed; "
                 "Protocol A eval n=4,200 per seed; exec_fail counted wrong.")
write_sheet(wb, "Significance",
            ["Comparison", "dAUROC", "CI95 lo", "CI95 hi", "p(d<=0)",
             "dselEX@30", "CI95 lo", "CI95 hi", "p(d<=0)"],
            [[nm, a["delta"], a["ci95"][0], a["ci95"][1], a["p_le0"],
              s["delta"], s["ci95"][0], s["ci95"][1], s["p_le0"]]
             for nm, (a, s) in pairs.items()],
            note="Question-level cluster bootstrap, 2,000 resamples, "
                 "seed-0 Protocol A (n=4,200).")
write_sheet(wb, "Ablation",
            ["B4 variant", "AUROC", "ECE15", "selEX@50", "selEX@30"],
            [[nm, m["auroc"], m["ece"], m["s50"], m["s30"]]
             for nm, m in abl.items()],
            note="Seed-0 split. Importance (|std. coef| x fit-set sd, desc): "
                 + "; ".join(f"{nm}={v:.2f}" for v, nm in importance))
write_sheet(wb, "ExecFailSens",
            ["Method", "Protocol", "n", "AUROC", "ECE15", "selEX@50",
             "selEX@30"],
            [[nm, "exec_fail=wrong (main)", a["n"], a["auroc"], a["ece"],
              a["s50"], a["s30"]] for nm, (a, b) in sens.items()]
            + [[nm, "exec_fail excluded", b["n"], b["auroc"], b["ece"],
                b["s50"], b["s30"]] for nm, (a, b) in sens.items()],
            note="Seed-0 Protocol A. Excluded protocol: correct+silent only; "
                 "B0 selEX equals always-execute EX on the kept set.")
lay_rows = [[strat, "holds/5", mond[strat]["holds"], None, None, None, None]
            for strat in ("global", "difficulty", "score")]
lay_rows += [r for strat in ("global", "difficulty", "score")
             for r in mond[strat]["layers"]]
write_sheet(wb, "MondrianLayers",
            ["Stratify", "seed", "Layer", "n_cal", "n_cal_wrong", "n_test",
             "n_test_wrong"],
            lay_rows,
            note="B8 on B4, alpha=0.02, cal=40% questions, seeds 0-4. "
                 "'holds/5' rows verify paper Table 6 (2/5, 4/5, 4/5).")
wb.save(xlp)
print(f"[out] {xlp} (+5 sheets)", flush=True)
print("ALL DONE", flush=True)
