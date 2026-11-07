# -*- coding: utf-8 -*-
"""run_chal_expand.py — challenging 层扩池验证（第二轮修订 R3）。

动机: 主池 300 题中 challenging 仅 28 题（seed-0 Protocol A eval 内 20 题/400
候选），B4 在该层 AUROC≈0.502（chance）。无法区分"样本量不足"与"真实失效"。

方案: BIRD dev 145 道 challenging 中原池未覆盖的 117 道（chal117_qids.json），
用与主池完全相同的管线（make_pool.py --logprobs, deepseek-chat,
nucleus10+beam10, 严格结果集标注）生成扩池后评测:
  - B4（13 子句特征 + z, logistic）在旧池 30% 题级 fit（seeds 0-4），
    在全部新 challenging 题上评测（新题与旧池零重叠，无泄漏）；
  - 同表报 z-only / share-only(B2 复现) / fuse_prod；
  - 对照行: 旧池 eval split 内 challenging 层的 B4（同协议同 seed）；
  - AUROC 题目级聚类 bootstrap（2000 次）95% CI 与配对 Δ(fuse_prod−B4)，
    并检验 B4 AUROC 是否仍与 0.5 相容。

前置校验: B2 share 计算在旧池上必须逐位复现已发布 b2 jsonl。

输出: results/chal_expand.md + paper/results/main_table.xlsx 两张新 sheet
（ChalExpand / ChalExpandBoot）。
"""
from __future__ import annotations

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
from gen_excel import load_conf, write_sheet  # noqa: E402

import openpyxl

POOLDIR = os.path.join(_ROOT, "data", "pools")
RESDIR = os.path.join(_ROOT, "results")
XLDIR = os.path.join(_ROOT, "results")
NEW_PREFIX = "bird117chal_ds"
SEEDS = [0, 1, 2, 3, 4]


def log(msg):
    print(msg, flush=True)


def sig(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


# ---------------- 数据加载 ----------------
def load_prefix(prefix):
    pool = load_pool(os.path.join(POOLDIR, prefix + "_pool.jsonl"),
                     require_logprobs=True)
    labels = load_labels(os.path.join(POOLDIR, prefix + "_labels.jsonl"))
    rows = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue
        base = math.exp(sum(c.token_logprobs))
        z = math.log(max(base, 1e-12))   # 与 run_b4_recalib 一致（含截断）
        rows.append((c, z, clause_features(c.sql), lab["correct"], lab["raw"]))
    return rows, labels


def b2_shares(rows):
    """复现 run_b2.py: 每 (qid, source) 组内 k=10, conf=多数标签占比
    (仅 correct/silent 有效; 全失败=0.0)。组大小=k 时不采样, 确定性。"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r[0].qid, r[0].source)].append(r)
    conf = {}
    for (_, _), cands in sorted(groups.items()):
        valid = [r[4] for r in cands if r[4] in ("correct", "silent")]
        v = 0.0 if not valid else \
            max(valid.count("correct"), valid.count("silent")) / len(valid)
        for r in cands:
            conf[r[0].key] = v
    return conf


old_rows, labels_old = load_prefix("bird300_ds")
new_rows, labels_new = load_prefix(NEW_PREFIX)
log(f"[data] old rows={len(old_rows)} new rows={len(new_rows)}")
labmaps = {"old": {r[0].key: r for r in old_rows},
           "new": {r[0].key: r for r in new_rows}}

# ---------------- B2 share 复现校验（旧池 vs 已发布 jsonl） ----------------
shares_old = b2_shares(old_rows)
pub = {}
for fn in ("bird300_ds_b2_nuc10.jsonl", "bird300_ds_b2_beam10.jsonl"):
    pub.update(load_conf(fn))
d_share = max(abs(shares_old[k] - pub[k]) for k in pub if k in shares_old)
log(f"[check] B2 share reproduction max|d| = {d_share:.2e} "
    f"(n={sum(1 for k in pub if k in shares_old)})")
if d_share > 1e-9:
    raise SystemExit("[FATAL] B2 share 复现失败，停止")

shares_new = b2_shares(new_rows)

old_qids = sorted({r[0].qid for r in old_rows})
N_FIT = max(1, int(len(old_qids) * 0.3))
new_qids = sorted({r[0].qid for r in new_rows})
log(f"[data] old qids={len(old_qids)} (fit {N_FIT}/seed)  "
    f"new challenging qids={len(new_qids)}")


def split(seed):
    q = old_qids[:]
    random.Random(seed).shuffle(q)
    return set(q[:N_FIT])


# ---------------- B4 拟合（旧池 fit split, 复现原协议） ----------------
def fit_b4(fit_rows, mode):
    """mode: full(13标准化+z/10) / clause(13标准化) / z(z/10)。
    返回 score(row) — row=(c, z, feats, correct, raw)。"""
    dim = 13 if mode in ("full", "clause") else 0
    mu = [sum(r[2][j] for r in fit_rows) / len(fit_rows) for j in range(dim)]
    sd = [max(1e-6, math.sqrt(sum((r[2][j] - mu[j]) ** 2 for r in fit_rows)
                                 / len(fit_rows))) for j in range(dim)]

    def x_of(r):
        if mode == "full":
            return ([(f - m) / s for f, m, s in zip(r[2], mu, sd)]
                    + [r[1] / 10.0])
        if mode == "clause":
            return [(f - m) / s for f, m, s in zip(r[2], mu, sd)]
        return [r[1] / 10.0]

    w = fit_logistic([x_of(r) for r in fit_rows],
                     [1.0 if r[3] else 0.0 for r in fit_rows])

    def score(r):
        x = x_of(r)
        return sig(w[-1] + sum(wj * xj for wj, xj in zip(w[:-1], x)))
    return score


# ---------------- 指标（与 revision_round1 相同口径） ----------------
def sel_ex(cf, keys, cov, labmap):
    ranked = sorted(keys, key=lambda k: (-cf[k], k))
    take = ranked[:max(1, int(len(ranked) * cov))]
    return sum(labmap[k][3] for k in take) / len(take)


def eval_metrics(cf, keys, labmap):
    co = [labmap[k][3] for k in keys]
    sc = [cf[k] for k in keys]
    return {"auroc": M.auroc(sc, co), "ece": M.ece(sc, co),
            "s50": sel_ex(cf, keys, 0.5, labmap),
            "s30": sel_ex(cf, keys, 0.3, labmap),
            "n": len(keys), "ex": sum(co) / len(keys)}


# ---------------- 多 seed 评测 ----------------
multi = defaultdict(list)     # (evalset, method) -> [metrics per seed]
for seed in SEEDS:
    fit_qids = split(seed)
    fit_rows = [r for r in old_rows if r[0].qid in fit_qids]
    score_full = fit_b4(fit_rows, "full")
    score_z = fit_b4(fit_rows, "z")

    # 新 challenging 池（全部 117 题, 与 fit 零重叠）
    keys_new = sorted(r[0].key for r in new_rows)
    cf_b4 = {r[0].key: score_full(r) for r in new_rows}
    cf_z = {r[0].key: score_z(r) for r in new_rows}
    cf_sh = {r[0].key: shares_new.get(r[0].key, 0.5) for r in new_rows}
    cf_fp = {k: cf_b4[k] * shares_new.get(k, 0.5) for k in keys_new}
    for nm, cf in (("B4 Clause Recalib", cf_b4), ("z-only", cf_z),
                   ("Share-only (B2)", cf_sh), ("Fuse-prod", cf_fp)):
        multi[("new", nm)].append(eval_metrics(cf, keys_new, labmaps["new"]))

    # 对照: 旧池 eval split 内 challenging 层（同协议同 seed）
    keys_oc = sorted(r[0].key for r in old_rows
                     if r[0].qid not in fit_qids and r[0].difficulty == "challenging")
    cf_oc = {r[0].key: score_full(r) for r in old_rows
             if r[0].qid not in fit_qids and r[0].difficulty == "challenging"}
    multi[("old-chal", "B4 Clause Recalib")].append(
        eval_metrics(cf_oc, keys_oc, labmaps["old"]))
    log(f"[seed {seed}] new n={len(keys_new)} old-chal n={len(keys_oc)} "
        f"B4 new={multi[('new', 'B4 Clause Recalib')][-1]['auroc']:.4f} "
        f"B4 old-chal={multi[('old-chal', 'B4 Clause Recalib')][-1]['auroc']:.4f}")

# 新池统计
lab_cnt = defaultdict(int)
for r in new_rows:
    lab_cnt[r[4]] += 1
log(f"[stats] new pool labels: {dict(lab_cnt)}, "
    f"EX={lab_cnt['correct'] / len(new_rows):.4f}")

# ---------------- 题目级聚类 bootstrap（新池） ----------------
cf_b4_s0, cf_fp_s0 = {}, {}
fit_rows0 = [r for r in old_rows if r[0].qid in split(0)]
score_full0 = fit_b4(fit_rows0, "full")
for r in new_rows:
    cf_b4_s0[r[0].key] = score_full0(r)
    cf_fp_s0[r[0].key] = cf_b4_s0[r[0].key] * shares_new.get(r[0].key, 0.5)
q2keys = defaultdict(list)
for r in new_rows:
    q2keys[r[0].qid].append(r[0].key)
qlist = sorted(q2keys)

rng = random.Random(0)
draws = {"b4": [], "fp": [], "d": []}
n_skip = 0
for _ in range(2000):
    qs = [rng.choice(qlist) for _ in qlist]
    ks = [k for q in qs for k in q2keys[q]]
    co = [labmaps["new"][k][3] for k in ks]
    if 0 < sum(co) < len(co):
        ab4 = M.auroc([cf_b4_s0[k] for k in ks], co)
        afp = M.auroc([cf_fp_s0[k] for k in ks], co)
        draws["b4"].append(ab4)
        draws["fp"].append(afp)
        draws["d"].append(afp - ab4)
    else:
        n_skip += 1


def ci(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v))]


lo_b4, hi_b4 = ci(draws["b4"])
lo_fp, hi_fp = ci(draws["fp"])
lo_d, hi_d = ci(draws["d"])
p_d = sum(1 for x in draws["d"] if x <= 0) / len(draws["d"])
p_b4 = sum(1 for x in draws["b4"] if x <= 0.5) / len(draws["b4"])
log(f"[boot] B4 AUROC CI [{lo_b4:.4f},{hi_b4:.4f}]  FP [{lo_fp:.4f},{hi_fp:.4f}]"
    f"  d [{lo_d:+.4f},{hi_d:+.4f}] p(d<=0)={p_d:.3f}  p(B4<=0.5)={p_b4:.3f}"
    f"  skip={n_skip}")

# ---------------- 输出 markdown ----------------
os.makedirs(RESDIR, exist_ok=True)
def ms_str(vals, k):
    xs = [v[k] for v in vals]
    return f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}"
lines = [
    "# Challenging 层扩池验证（R3）",
    "",
    "协议: B4 在旧池 300 题的 30% 题级 fit（seeds 0-4），在全部 117 道新 challenging 题",
    "（与旧池零重叠）上评测；exec_fail 计 wrong，确定性平局；与主表口径完全一致。",
    "扩池管线与主池完全相同（make_pool.py --logprobs, deepseek-chat,",
    "nucleus10+beam10, 严格结果集标注）。",
    "",
    "## 新池统计",
    "",
    f"- 候选数 {len(new_rows)}（117 题 × 20），题目级 EX = "
    f"{lab_cnt['correct'] / len(new_rows):.4f}",
    f"- 标签: correct {lab_cnt['correct']} / silent {lab_cnt['silent']} / "
    f"exec_fail {lab_cnt['exec_fail']}",
    f"- B2 share 复现校验（旧池 vs 已发布 jsonl）: max|d| = {d_share:.2e}",
    "",
    "## 主表（新 challenging 池, seeds 0-4 mean±std）",
    "",
    "| Method | AUROC | ECE15 | selEX@50 | selEX@30 |",
    "|---|---|---|---|---|",
]
for nm in ("B4 Clause Recalib", "z-only", "Share-only (B2)", "Fuse-prod"):
    vals = multi[("new", nm)]
    lines.append(f"| {nm} | " + " | ".join(ms_str(vals, k) for k in
                                           ("auroc", "ece", "s50", "s30")) + " |")
vals = multi[("old-chal", "B4 Clause Recalib")]
lines += [
    f"| [对照] B4 on 旧池 eval-challenging | " +
    " | ".join(ms_str(vals, k) for k in ("auroc", "ece", "s50", "s30")) + " |",
    "",
    "各 seed 明细（AUROC）:",
    "",
    "| Method | " + " | ".join(f"s{s}" for s in SEEDS) + " |",
    "|---|" + "---|" * len(SEEDS),
]
for nm in ("B4 Clause Recalib", "z-only", "Share-only (B2)", "Fuse-prod"):
    lines.append(f"| {nm} (new) | " +
                 " | ".join(f"{v['auroc']:.4f}" for v in multi[("new", nm)]) + " |")
lines.append("| [对照] B4 (old eval-chal) | " +
             " | ".join(f"{v['auroc']:.4f}"
                        for v in multi[("old-chal", "B4 Clause Recalib")]) + " |")
lines += [
    "",
    "## 题目级聚类 bootstrap（seed-0 fit, 2000 次）",
    "",
    f"- B4 AUROC 95% CI [{lo_b4:.4f}, {hi_b4:.4f}]，"
    f"P(AUROC<=0.5)={p_b4:.3f}",
    f"- Fuse-prod AUROC 95% CI [{lo_fp:.4f}, {hi_fp:.4f}]",
    f"- Δ(fuse_prod−B4) [{lo_d:+.4f}, {hi_d:+.4f}]，p(Δ<=0)={p_d:.3f}",
    "",
]
with open(os.path.join(RESDIR, "chal_expand.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log(f"[out] {os.path.join(RESDIR, 'chal_expand.md')}")

# ---------------- 输出 xlsx ----------------
xl_path = os.path.join(XLDIR, "main_table.xlsx")
wb = openpyxl.load_workbook(xl_path)
hdr = ["Method", "Eval set", "AUROC", "ECE15", "selEX@50", "selEX@30", "n"]
rows = []
for nm, eset in (("B4 Clause Recalib", "new challenging (n=2340)"),
                 ("z-only", "new challenging (n=2340)"),
                 ("Share-only (B2)", "new challenging (n=2340)"),
                 ("Fuse-prod", "new challenging (n=2340)"),
                 ("B4 Clause Recalib", "old eval-challenging (control)")):
    vals = multi[("new" if eset.startswith("new") else "old-chal", nm)]
    rows.append([nm, eset] + [ms_str(vals, k) for k in
                              ("auroc", "ece", "s50", "s30")] +
                [st.mean(v["n"] for v in vals)])
note = ("B4 fit on 30% of old 300 questions per seed (0-4); evaluated on all "
        "117 new challenging questions (zero overlap, no leakage). Same "
 "protocol as main table (exec_fail counted wrong).")
write_sheet(wb, "ChalExpand", hdr, rows, note=note)
hdr2 = ["Metric", "Estimate", "95% CI low", "95% CI high", "p"]
rows2 = [
    ["B4 AUROC (new chal, seed-0)", f"{st.mean(draws['b4']):.4f}",
     f"{lo_b4:.4f}", f"{hi_b4:.4f}", f"P(AUROC<=0.5)={p_b4:.3f}"],
    ["Fuse-prod AUROC (new chal, seed-0)", f"{st.mean(draws['fp']):.4f}",
     f"{lo_fp:.4f}", f"{hi_fp:.4f}", ""],
    ["Delta (fuse_prod - B4) AUROC", f"{st.mean(draws['d']):+.4f}",
     f"{lo_d:+.4f}", f"{hi_d:+.4f}", f"p(Delta<=0)={p_d:.3f}"],
]
write_sheet(wb, "ChalExpandBoot", hdr2, rows2,
            note="Question-level cluster bootstrap, 2000 draws, seed-0 fit.")
wb.save(xl_path)
log(f"[out] {xl_path} (+2 sheets)")
log("ALL DONE")
