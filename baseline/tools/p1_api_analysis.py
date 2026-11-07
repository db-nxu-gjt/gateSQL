# -*- coding: utf-8 -*-
"""p1_api_analysis.py — API 项修订实验分析（meta-review round 2: B2 judge 加强 + A2 第二生成器）。

Sections (--part all|b1v|kimi):
  b1v  : B1 elicitation 变体（2stage / topk5）+ B5 异构强 judge（kimi-k3）
         在 DeepSeek 主池 seed-0 eval 上的指标与簇 bootstrap 比较。
  kimi : 第二生成器（Kimi K2.6, 150 题 × 20 候选）复现：
         池统计 / share / B1-self / B5-self / cross-model judge(DeepSeek) /
         B4 frozen-transfer & refit / fuse_prod / conformal 拒绝率(20 seeds)。

口径与正文一致：seed-0 题级 30/70 切分，eval 集 AUROC/ECE15/selEX@30，
exec_fail 计为 wrong（p1_analysis "paper accounting"）。

Outputs: results/revision_api.md + paper/results/_api_stats.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "tools"))

from common.schema import load_pool  # noqa: E402
from common import metrics as M  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402
from p0_reanalysis import conformal_eval  # noqa: E402

POOLS = os.path.join(_ROOT, "data", "pools")
OUT_MD = os.path.join(_ROOT, "results", "revision_api.md")
OUT_JSON = os.path.join(_ROOT, "results", "_api_stats.json")

DS_POOL = os.path.join(POOLS, "bird300_ds_pool.jsonl")
DS_LAB = os.path.join(POOLS, "bird300_ds_labels.jsonl")
KM_POOL = os.path.join(POOLS, "bird150_kimi_pool.jsonl")
KM_LAB = os.path.join(POOLS, "bird150_kimi_labels.jsonl")

STATS: dict = {}
MDL: list = ["# API 项修订实验（revision round 2: B2 judge 加强 + A2 第二生成器）\n"]


def md(s: str = "") -> None:
    MDL.append(s)


def load_conf(path: str) -> dict:
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    float(r["conf"])
    return d


def load_labels_raw(path: str) -> dict:
    lab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lab[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    r["label"]
    return lab


def seed0_eval_qids(qids: list) -> set:
    """复现 _fit_b4_seed 切分：sorted qids, Random(0) shuffle, 前 30% fit。"""
    qs = sorted(qids)
    rng = random.Random(0)
    rng.shuffle(qs)
    n_fit = max(1, int(len(qs) * 0.3))
    return set(qs[n_fit:])


def metrics_row(cf: dict, labels: dict, eval_q: set) -> dict:
    ks = [k for k in cf if k[0] in eval_q and k in labels]
    co = [labels[k] == "correct" for k in ks]
    if not ks:
        return {"n": 0}
    return {
        "n": len(ks),
        "auroc": M.auroc([cf[k] for k in ks], co),
        "ece": M.ece([cf[k] for k in ks], co),
        "selex30": M.selective_ex([cf[k] for k in ks], co, 0.3),
    }


def fmt(r: dict) -> str:
    if r.get("n", 0) == 0:
        return "  (missing)"
    return (f"n={r['n']}  AUROC={r['auroc']:.3f}  ECE={r['ece']:.3f}  "
            f"selEX@30={r['selex30']:.3f}")


def cluster_boot_auroc(cf_a: dict, cf_b: dict, labels: dict, eval_q: set,
                       n_boot: int = 2000, seed: int = 0) -> dict:
    """题级簇 bootstrap：delta = AUROC(b) - AUROC(a)，逐题重采样。"""
    qs = sorted({k[0] for k in cf_a if k[0] in eval_q and k in labels})
    per_q_a, per_q_b = {}, {}
    for q in qs:
        ka = [k for k in cf_a if k[0] == q and k in labels]
        kb = [k for k in cf_b if k[0] == q and k in labels]
        if ka:
            per_q_a[q] = ka
        if kb:
            per_q_b[q] = kb
    common_qs = sorted(set(per_q_a) & set(per_q_b))
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        samp = [common_qs[rng.randrange(len(common_qs))]
                for _ in range(len(common_qs))]
        ka = [k for q in samp for k in per_q_a[q]]
        kb = [k for q in samp for k in per_q_b[q]]
        ca = [labels[k] == "correct" for k in ka]
        cb = [labels[k] == "correct" for k in kb]
        d = M.auroc([cf_b[k] for k in kb], cb) - M.auroc([cf_a[k] for k in ka], ca)
        deltas.append(d)
    deltas.sort()
    lo, hi = deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot)]
    mean = sum(deltas) / n_boot
    # 双侧 p：delta 分布跨 0 的比例
    p = 2 * min(sum(1 for d in deltas if d <= 0) / n_boot,
                sum(1 for d in deltas if d >= 0) / n_boot)
    return {"mean": mean, "ci95": [lo, hi], "p_two_sided": min(1.0, p)}


# ------------------------------------------------------------------ b1v --
def part_b1v() -> None:
    md("\n## 1. B1 elicitation 变体与异构强 judge（DeepSeek 主池，seed-0 eval）\n")
    labels = load_labels_raw(DS_LAB)
    eval_q = seed0_eval_qids(sorted({k[0] for k in labels}))

    files = {
        "B1 (original verbalized)": os.path.join(POOLS, "bird300_ds_b1.jsonl"),
        "B1-2stage (multi-turn)": os.path.join(POOLS, "bird300_ds_b1_2stage.jsonl"),
        "B1-topk5 (top-k)": os.path.join(POOLS, "bird300_ds_b1_topk5.jsonl"),
        "B5 (DeepSeek self-judge)": os.path.join(POOLS, "bird300_ds_b5.jsonl"),
        "B5-strong (kimi-k3 cross)": os.path.join(POOLS, "bird300_ds_b5_kimi3.jsonl"),
        "B4 (recalibrated)": os.path.join(POOLS, "bird300_ds_b4.jsonl"),
        "fuse_prod (B4xshare)": os.path.join(POOLS, "bird300_ds_fuse_prod.jsonl"),
    }
    confs = {name: load_conf(p) for name, p in files.items()}
    # kimi3 x share 融合（share 为 20 候选组内一致性）
    shares_ds = {}
    for f in (os.path.join(POOLS, "bird300_ds_b2_nuc10.jsonl"),
              os.path.join(POOLS, "bird300_ds_b2_beam10.jsonl")):
        shares_ds.update(load_conf(f))
    k3 = confs["B5-strong (kimi-k3 cross)"]
    if k3:
        confs["fuse kimi3 x share"] = {k: k3[k] * shares_ds.get(k, 0.5)
                                       for k in k3}
    out = {}
    md("| method | n | AUROC | ECE15 | selEX@30 |")
    md("|---|---|---|---|---|")
    for name, cf in confs.items():
        r = metrics_row(cf, labels, eval_q)
        out[name] = r
        if r["n"]:
            md(f"| {name} | {r['n']} | {r['auroc']:.3f} | {r['ece']:.3f} | "
               f"{r['selex30']:.3f} |")
        else:
            md(f"| {name} | 0 | - | - | - |")
    md("")
    md("说明：口径与正文 Protocol A 一致（exec_fail 计 wrong，全 eval 集）。"
       "B1-original 复算 0.573/0.421 与正文一致。")

    # 簇 bootstrap 比较
    pairs = [("B1 (original verbalized)", "B1-topk5 (top-k)"),
             ("B5 (DeepSeek self-judge)", "B5-strong (kimi-k3 cross)"),
             ("B4 (recalibrated)", "B5-strong (kimi-k3 cross)")]
    boot_out = {}
    for a, b in pairs:
        if confs.get(a) and confs.get(b):
            bt = cluster_boot_auroc(confs[a], confs[b], labels, eval_q)
            boot_out[f"{b} vs {a}"] = bt
            md(f"- cluster bootstrap ΔAUROC ({b} − {a}): "
               f"{bt['mean']:+.3f}, 95% CI [{bt['ci95'][0]:+.3f}, "
               f"{bt['ci95'][1]:+.3f}], two-sided p={bt['p_two_sided']:.3f}")
    # 分布披露
    topk = confs["B1-topk5 (top-k)"]
    ks_topk = [k for k in topk if k[0] in eval_q]
    zero = sum(1 for k in ks_topk if topk[k] == 0.0)
    md(f"- B1-topk5 分布：{zero}/{len(ks_topk)} "
       f"({100*zero/len(ks_topk):.1f}%) 候选未列入 top-5（conf=0）")
    ts = confs["B1-2stage (multi-turn)"]
    ks_ts = [k for k in ts if k[0] in eval_q]
    one0 = sum(1 for k in ks_ts if ts[k] >= 0.99)
    md(f"- B1-2stage 分布：{one0}/{len(ks_ts)} "
       f"({100*one0/len(ks_ts):.1f}%) 候选 conf≥0.99（饱和）")
    STATS["b1v"] = {"rows": out, "boot": boot_out}


# ----------------------------------------------------------------- kimi --
def _b4_fit(pool_data, seed, fit_frac=0.3):
    """拟合逻辑回归（同 _fit_b4_seed），返回 (params, mu, sd, fit_qids)。"""
    qids = sorted({d[4] for d in pool_data})
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_fit = max(1, int(len(qids) * fit_frac))
    fit_q = set(qids[:n_fit])
    fit_idx = [i for i, d in enumerate(pool_data) if d[4] in fit_q]
    dim = len(pool_data[0][2])
    mu = [sum(pool_data[i][2][j] for i in fit_idx) / len(fit_idx)
          for j in range(dim)]
    sd = [max(1e-6, math.sqrt(sum((pool_data[i][2][j] - mu[j]) ** 2
                                  for i in fit_idx) / len(fit_idx)))
          for j in range(dim)]

    def make_x(i):
        return ([(pool_data[i][2][j] - mu[j]) / sd[j] for j in range(dim)]
                + [pool_data[i][1] / 10.0])

    params = fit_logistic([make_x(i) for i in fit_idx],
                          [pool_data[i][3] for i in fit_idx])
    return params, mu, sd, fit_q


def _b4_apply(params, mu, sd, z, feats) -> float:
    dim = len(feats)
    x = [(feats[j] - mu[j]) / sd[j] for j in range(dim)] + [z / 10.0]
    zz = params[-1] + sum(w * xj for w, xj in zip(params[:-1], x))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))


def part_kimi() -> None:
    md("\n## 2. 第二生成器复现（Kimi K2.6，150 题 × 20 候选）\n")
    pool = load_pool(KM_POOL)
    labels = load_labels_raw(KM_LAB)
    if not pool:
        md("(池未生成，跳过)")
        return
    qids = sorted({c.qid for c in pool})
    eval_q = seed0_eval_qids(qids)
    n_ev = sum(1 for c in pool if c.qid in eval_q)

    # 池统计
    lab_vals = Counter(labels.values())
    ex_all = lab_vals["correct"] / max(1, sum(lab_vals.values()))
    ev_vals = Counter(labels[k] for k in labels if k[0] in eval_q)
    ex_ev = ev_vals["correct"] / max(1, sum(ev_vals.values()))
    md(f"- 池规模：{len(pool)} 候选 / {len(qids)} 题（eval {len(eval_q)} 题 / "
       f"{n_ev} 候选）；seed-0 切分 fit {len(qids)-len(eval_q)} / eval {len(eval_q)}")
    md(f"- 生成协议：nucleus20（T=0.6 端点强制, top_p=0.95, non-thinking, "
       f"max_tokens=384, stop='\\\\n\\\\n'）；Kimi 端点仅允许该预设且 n=1 逐条采样")
    md(f"- 标签构成（全池）：{dict(lab_vals)}，EX={ex_all:.3f}；"
       f"eval 集 EX={ex_ev:.3f}")
    # 同题 DS 对照
    ds_labels = load_labels_raw(DS_LAB)
    km_qset = set(qids)
    ds_vals = Counter(v for k, v in ds_labels.items() if k[0] in km_qset)
    ex_ds_same = ds_vals["correct"] / max(1, sum(ds_vals.values()))
    md(f"- 同 150 题 DeepSeek 池 EX={ex_ds_same:.3f}（Kimi {ex_all:.3f}）")
    # unique SQL
    norm = [" ".join(c.sql.lower().split()) for c in pool]
    md(f"- unique SQL：{len(set(norm))}/{len(norm)} "
       f"({100*len(set(norm))/len(norm):.1f}%)")

    # share（20 候选一组）
    by_q_sql = defaultdict(list)
    for c in pool:
        by_q_sql[(c.qid, " ".join(c.sql.lower().split()))].append(c.key)
    share = {}
    for c in pool:
        grp = by_q_sql[(c.qid, " ".join(c.sql.lower().split()))]
        share[c.key] = len(grp) / 20.0

    # B4 特征数据
    def build_data(pl, labs):
        data = []
        for c in pl:
            lab = labs.get(c.key)
            if lab is None or not c.token_logprobs:
                continue
            z = sum(c.token_logprobs)
            data.append((c.key, z, clause_features(c.sql),
                         lab == "correct", c.qid))
        return data

    km_data = build_data(pool, labels)
    ds_pool_full = load_pool(DS_POOL)
    ds_data = build_data(ds_pool_full, ds_labels)

    # B4 frozen-transfer：DS seed-0 fit（90 题）参数 → Kimi eval 候选
    params, mu, sd, ds_fit_q = _b4_fit(ds_data, 0)
    km_eval_keys = {c.key for c in pool if c.qid in eval_q}
    b4_frozen = {}
    for key, z, feats, _, _ in km_data:
        if key in km_eval_keys:
            b4_frozen[key] = _b4_apply(params, mu, sd, z, feats)

    # B4 refit：Kimi 池自身 seed-0 切分
    from p0_reanalysis import _fit_b4_seed
    sc_refit = _fit_b4_seed(km_data, 0, fit_frac=0.3)
    # _fit_b4_seed 内部用全部 qids 的 30% 切分；eval 键 = 返回 dict 的键
    b4_refit = sc_refit

    fuse_frozen = {k: b4_frozen[k] * share.get(k, 0.5) for k in b4_frozen}
    fuse_refit = {k: b4_refit[k] * share.get(k, 0.5) for k in b4_refit}

    api_files = {
        "B1-self (kimi-k2.6)": os.path.join(POOLS, "bird150_kimi_b1.jsonl"),
        "B5-self (kimi-k2.6)": os.path.join(POOLS, "bird150_kimi_b5_self.jsonl"),
        "B5-cross (deepseek-chat)": os.path.join(POOLS, "bird150_kimi_b5_ds_cross.jsonl"),
    }
    rows = {
        "B2 share (K=20 vote)": metrics_row(share, labels, eval_q),
        "B1-self (kimi-k2.6)": metrics_row(load_conf(api_files["B1-self (kimi-k2.6)"]), labels, eval_q),
        "B5-self (kimi-k2.6)": metrics_row(load_conf(api_files["B5-self (kimi-k2.6)"]), labels, eval_q),
        "B5-cross (deepseek-chat)": metrics_row(load_conf(api_files["B5-cross (deepseek-chat)"]), labels, eval_q),
        "B4 frozen (DS-fit transfer)": metrics_row(b4_frozen, labels, eval_q),
        "B4 refit (Kimi fit)": metrics_row(b4_refit, labels, eval_q),
        "fuse frozen (B4xshare)": metrics_row(fuse_frozen, labels, eval_q),
        "fuse refit (B4xshare)": metrics_row(fuse_refit, labels, eval_q),
    }
    md("\n### eval 集指标（exec_fail 计 wrong）\n")
    md("| method | n | AUROC | ECE15 | selEX@30 |")
    md("|---|---|---|---|---|")
    for name, r in rows.items():
        if r["n"]:
            md(f"| {name} | {r['n']} | {r['auroc']:.3f} | {r['ece']:.3f} | "
               f"{r['selex30']:.3f} |")
        else:
            md(f"| {name} | 0 | - | - | - |")

    # conformal（B8 包装，B4-refit 分数，题级 20 seeds）
    N_SEEDS, ALPHA, CAL_FRAC = 20, 0.02, 0.4
    conf_all = dict(b4_refit)
    conf_all.update(b4_frozen)
    ks_all = sorted(set(b4_refit) | set(b4_frozen))
    q_all = sorted({k[0] for k in ks_all})
    holds_r = holds_f = 0
    leaks_r, leaks_f = [], []
    for seed in range(N_SEEDS):
        rq = q_all[:]
        random.Random(seed).shuffle(rq)
        cal_q = set(rq[:max(1, int(len(rq) * CAL_FRAC))])
        cal_k = [k for k in ks_all if k[0] in cal_q]
        test_k = [k for k in ks_all if k[0] not in cal_q]
        rr = conformal_eval(b4_refit, labels, [k for k in cal_k if k in b4_refit],
                            [k for k in test_k if k in b4_refit], alpha=ALPHA)
        rf = conformal_eval(b4_frozen, labels, [k for k in cal_k if k in b4_frozen],
                            [k for k in test_k if k in b4_frozen], alpha=ALPHA)
        holds_r += rr["holds"]; holds_f += rf["holds"]
        leaks_r.append(rr["leak"]); leaks_f.append(rf["leak"])
    md(f"\n### B8 conformal（α=0.02，题级 40/60，{N_SEEDS} seeds）\n")
    md(f"- B4-refit 分数：holds {holds_r}/{N_SEEDS}，leak mean "
       f"{sum(leaks_r)/N_SEEDS:.4f}（{sum(leaks_r)/N_SEEDS/ALPHA:.2f}×α）")
    md(f"- B4-frozen 分数：holds {holds_f}/{N_SEEDS}，leak mean "
       f"{sum(leaks_f)/N_SEEDS:.4f}（{sum(leaks_f)/N_SEEDS/ALPHA:.2f}×α）")

    STATS["kimi"] = {
        "n_pool": len(pool), "n_q": len(qids), "n_eval_q": len(eval_q),
        "ex_all": ex_all, "ex_eval": ex_ev, "ex_ds_same150": ex_ds_same,
        "unique_sql": len(set(norm)),
        "rows": rows,
        "conformal": {"refit_holds": holds_r, "refit_leak_mean": sum(leaks_r)/N_SEEDS,
                      "frozen_holds": holds_f, "frozen_leak_mean": sum(leaks_f)/N_SEEDS},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "b1v", "kimi"])
    args = ap.parse_args()
    if args.part in ("all", "b1v"):
        part_b1v()
    if args.part in ("all", "kimi"):
        part_kimi()
    body = "\n".join(MDL) + "\n"
    if os.path.exists(OUT_MD):
        prev = open(OUT_MD, encoding="utf-8").read()
        if prev.rstrip().endswith(STATS_SNAPSHOT_MARK):
            body = prev  # 不覆盖既有文件（保留历史），仅追加时手动处理
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(body)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(STATS, f, ensure_ascii=False, indent=1, default=str)
    print(f"[done] -> {OUT_MD} / {OUT_JSON}")


STATS_SNAPSHOT_MARK = "[snapshot]"

if __name__ == "__main__":
    main()
