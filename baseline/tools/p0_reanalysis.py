# -*- coding: utf-8 -*-
"""p0_reanalysis.py — P0 revision reanalyses (meta-review round 1).

Sections (run: python p0_reanalysis.py --part all|conformal|perq|lodo|seeds|boot|misc):
  conformal : 20-seed B8 reproduction + question-level permutation KS-1/KS-2
              + candidate-level random-split control + synthetic exchangeable
              positive control (size validation).
  perq      : per-question top-1 selective accuracy / pass@1 / single-sample
              baselines / abstention composition (deployment semantics).
  lodo      : reproduce LODO per-fold fits; matched global-fit comparison on
              identical held-out eval candidates (protocol fix for A4).
  seeds     : 5-refit-seed spread of selEX@30 for B4 and fuse_prod.
  boot      : question-level cluster bootstrap for the reported pairwise tests
              (consistent two-sided p + percentile CI).
  misc      : high-end purity thresholds w/ CP intervals, question-level
              bootstrap upper bound on realized leak (seed 0), isotonic
              tie-aware ranking metrics, pool composition vs full BIRD dev,
              unique-SQL counts per decoding group.

Outputs: results/revision_p0_reanalysis.md + paper/results/_p0_stats.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))

from common.schema import load_pool  # noqa: E402
from common import metrics as M  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402

POOLS = os.path.join(_ROOT, "data", "pools")
LAB_F = os.path.join(POOLS, "bird300_ds_labels.jsonl")
POOL_F = os.path.join(POOLS, "bird300_ds_pool.jsonl")
B4_F = os.path.join(POOLS, "bird300_ds_b4.jsonl")
FUSE_F = os.path.join(POOLS, "bird300_ds_fuse_prod.jsonl")
B1_F = os.path.join(POOLS, "bird300_ds_b1.jsonl")
ISO_F = os.path.join(POOLS, "bird300_ds_b4_iso.jsonl")
SH_F = [os.path.join(POOLS, "bird300_ds_b2_nuc10.jsonl"),
        os.path.join(POOLS, "bird300_ds_b2_beam10.jsonl")]
DEV_F = os.path.join(_ROOT, "data", "raw", "bird", "dev.json")

ALPHA = 0.02
CAL_FRAC = 0.4
N_SEEDS = 20
OUT_MD = os.path.join(_ROOT, "results", "revision_p0_reanalysis.md")
OUT_JSON = os.path.join(_ROOT, "results", "_p0_stats.json")

STATS: dict = {}
MD: list[str] = ["# P0 重分析（revision round 1）\n"]


def md(s: str = "") -> None:
    MD.append(s)


def load_labels() -> dict:
    lab = {}
    with open(LAB_F, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lab[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    r["label"]
    return lab


def load_conf(path: str) -> dict:
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    float(r["conf"])
    return d


def load_pool_meta() -> dict:
    """(qid,source,idx) -> {db, difficulty, sql}"""
    meta = {}
    with open(POOL_F, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                meta[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = {
                    "db": r.get("db_id", ""), "diff": r.get("difficulty", ""),
                    "sql": r.get("sql", ""),
                }
    return meta


def ks_2samp_d(a, b):
    from scipy import stats
    r = stats.ks_2samp(a, b)
    return r.statistic, r.pvalue


def ks_unif_d(vals):
    from scipy import stats
    r = stats.kstest(vals, "uniform")
    return r.statistic, r.pvalue


def conformal_eval(conf: dict, labels: dict, cal_keys, test_keys, alpha=ALPHA):
    """B8 semantics (run_b8_conformal.py): calibrate on wrong candidates of
    the cal set; p = (1+#{cal_wrong nonconf <= cand nonconf})/(n+1); accept
    for execution iff p<=alpha. leak = P(accept|wrong) (guaranteed quantity);
    coverage = fraction of test candidates accepted for execution."""
    cal_w = sorted(1.0 - conf[k] for k in cal_keys if labels[k] != "correct")
    test_w = [1.0 - conf[k] for k in test_keys if labels[k] != "correct"]
    n = len(cal_w)
    if n == 0 or not test_w:
        return None
    n_acc = acc_wrong = 0
    for k in test_keys:
        if (1 + bisect.bisect_right(cal_w, 1.0 - conf[k])) / (n + 1) <= alpha:
            n_acc += 1
            if labels[k] != "correct":
                acc_wrong += 1
    leak = acc_wrong / len(test_w)
    d1, p1 = ks_2samp_d(cal_w, test_w)
    pvals = [(1 + bisect.bisect_right(cal_w, s)) / (n + 1) for s in test_w]
    d2, p2 = ks_unif_d(pvals)
    return {"leak": leak, "holds": leak <= alpha + 1e-9,
            "coverage": n_acc / len(test_keys), "n_cal_wrong": n,
            "n_test_wrong": len(test_w), "ks1_d": d1, "ks1_p": p1,
            "ks2_d": d2, "ks2_p": p2}


def part_conformal():
    md("\n## 1. Conformal 重分析（A1）\n")
    labels = load_labels()
    out = {}
    for name, path in (("b4", B4_F), ("fuse_prod", FUSE_F)):
        conf = load_conf(path)
        keys = sorted(conf)
        qids = sorted({k[0] for k in keys})
        n_cal = int(len(qids) * CAL_FRAC)
        # (a) reproduce question-level protocol (ks_test.py)
        rows = []
        for seed in range(N_SEEDS):
            rq = qids[:]
            random.Random(seed).shuffle(rq)
            cal_q = set(rq[:n_cal])
            cal_k = [k for k in keys if k[0] in cal_q]
            test_k = [k for k in keys if k[0] not in cal_q]
            rows.append(conformal_eval(conf, labels, cal_k, test_k))
        leaks = [r["leak"] for r in rows]
        d1s = sorted(r["ks1_d"] for r in rows)
        d2s = sorted(r["ks2_d"] for r in rows)
        med = lambda v: v[len(v) // 2]
        out[name] = {
            "leak_mean": sum(leaks) / len(leaks),
            "leak_median": med(sorted(leaks)),
            "leak_max": max(leaks), "holds": sum(r["holds"] for r in rows),
            "ks1_reject_nominal": sum(r["ks1_p"] < 0.05 for r in rows),
            "ks2_reject_nominal": sum(r["ks2_p"] < 0.05 for r in rows),
            "ks1_d_median": med(d1s), "ks1_d_q1": d1s[len(d1s) // 4],
            "ks1_d_q3": d1s[3 * len(d1s) // 4],
            "ks2_d_median": med(d2s), "ks2_d_q1": d2s[len(d2s) // 4],
            "ks2_d_q3": d2s[3 * len(d2s) // 4],
        }
        # (b) question-level permutation test for KS-1/KS-2
        B = 200
        perm_rows = []
        for seed in range(N_SEEDS):
            rq = qids[:]
            random.Random(seed).shuffle(rq)
            cal_q = set(rq[:n_cal])
            cal_k = [k for k in keys if k[0] in cal_q]
            test_k = [k for k in keys if k[0] not in cal_q]
            obs = conformal_eval(conf, labels, cal_k, test_k)
            c1 = c2 = 0
            for perm in range(B):
                rq2 = qids[:]
                random.Random(10_000 + 77 * seed + perm).shuffle(rq2)
                cal_q2 = set(rq2[:n_cal])
                cal_k2 = [k for k in keys if k[0] in cal_q2]
                test_k2 = [k for k in keys if k[0] not in cal_q2]
                pp = conformal_eval(conf, labels, cal_k2, test_k2)
                if pp["ks1_d"] >= obs["ks1_d"]:
                    c1 += 1
                if pp["ks2_d"] >= obs["ks2_d"]:
                    c2 += 1
            perm_rows.append({"seed": seed,
                              "p_perm_ks1": (1 + c1) / (B + 1),
                              "p_perm_ks2": (1 + c2) / (B + 1),
                              "ks1_d": obs["ks1_d"], "ks2_d": obs["ks2_d"]})
        out[name]["perm_reject_ks1"] = sum(r["p_perm_ks1"] < 0.05
                                           for r in perm_rows)
        out[name]["perm_reject_ks2"] = sum(r["p_perm_ks2"] < 0.05
                                           for r in perm_rows)
        out[name]["perm_rows"] = perm_rows
        # (c) candidate-level random split control
        cand_rows = []
        for seed in range(N_SEEDS):
            rk = keys[:]
            random.Random(20_000 + seed).shuffle(rk)
            m = int(len(rk) * CAL_FRAC)
            cand_rows.append(conformal_eval(conf, labels, rk[:m], rk[m:]))
        cleaks = [r["leak"] for r in cand_rows]
        out[name]["cand_split"] = {
            "holds": sum(r["holds"] for r in cand_rows),
            "leak_mean": sum(cleaks) / len(cleaks),
            "leak_median": med(sorted(cleaks)), "leak_max": max(cleaks),
        }
        md(f"### {name}\n")
        r = out[name]
        md(f"- question-level 20 seeds: holds {r['holds']}/20, leak "
           f"mean {r['leak_mean']:.4f} ({r['leak_mean'] / ALPHA:.2f}xα), "
           f"median {r['leak_median']:.4f} ({r['leak_median'] / ALPHA:.2f}xα), "
           f"max {r['leak_max']:.4f} ({r['leak_max'] / ALPHA:.1f}xα)")
        md(f"- nominal KS: KS-1 reject {r['ks1_reject_nominal']}/20 "
           f"(D median {r['ks1_d_median']:.3f}, IQR "
           f"{r['ks1_d_q1']:.3f}-{r['ks1_d_q3']:.3f}); KS-2 reject "
           f"{r['ks2_reject_nominal']}/20 (D median {r['ks2_d_median']:.3f}, "
           f"IQR {r['ks2_d_q1']:.3f}-{r['ks2_d_q3']:.3f})")
        md(f"- **question-level permutation KS**: KS-1 reject "
           f"{r['perm_reject_ks1']}/20, KS-2 reject {r['perm_reject_ks2']}/20 "
           f"(B=200, cluster-preserving)")
        md(f"- **candidate-level random split control**: holds "
           f"{r['cand_split']['holds']}/20, leak mean "
           f"{r['cand_split']['leak_mean']:.4f} "
           f"({r['cand_split']['leak_mean'] / ALPHA:.2f}xα), max "
           f"{r['cand_split']['leak_max']:.4f} "
           f"({r['cand_split']['leak_max'] / ALPHA:.1f}xα)")

    # (d) synthetic exchangeable positive control (size validation)
    conf = load_conf(B4_F)
    keys = sorted(conf)
    wrong_nc = [1.0 - conf[k] for k in keys if labels[k] != "correct"]
    corr_nc = [1.0 - conf[k] for k in keys if labels[k] == "correct"]
    p_wrong = len(wrong_nc) / len(keys)

    def beta_mom(vals):
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        ab = m * (1 - m) / v - 1
        return max(1e-3, m * ab), max(1e-3, (1 - m) * ab)

    aw, bw = beta_mom(wrong_nc)
    ac, bc = beta_mom(corr_nc)
    syn_rows = []
    sconf = slab = klist = None
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        n_all = len(keys)
        n_cal = int(n_all * CAL_FRAC)
        sconf, slab = {}, {}
        for i in range(n_all):
            if rng.random() < p_wrong:
                sconf[("k", 0, i)] = 1.0 - rng.betavariate(aw, bw)
                slab[("k", 0, i)] = "silent"
            else:
                sconf[("k", 0, i)] = 1.0 - rng.betavariate(ac, bc)
                slab[("k", 0, i)] = "correct"
        klist = sorted(sconf)
        rng.shuffle(klist)
        syn_rows.append(conformal_eval(sconf, slab, klist[:n_cal],
                                       klist[n_cal:]))
    sleaks = [r["leak"] for r in syn_rows]
    out["synthetic_control"] = {
        "beta_wrong": [round(aw, 3), round(bw, 3)],
        "beta_correct": [round(ac, 3), round(bc, 3)],
        "auroc_check": round(M.auroc([sconf[k] for k in klist],
                                     [slab[k] == "correct" for k in klist]), 3),
        "holds": sum(r["holds"] for r in syn_rows),
        "leak_mean": sum(sleaks) / len(sleaks),
        "leak_max": max(sleaks),
        "ks1_reject": sum(r["ks1_p"] < 0.05 for r in syn_rows),
        "ks2_reject": sum(r["ks2_p"] < 0.05 for r in syn_rows),
        "n_seeds": N_SEEDS,
    }
    s = out["synthetic_control"]
    md("### synthetic exchangeable positive control (B4 marginals)\n")
    md(f"- i.i.d. Beta-fit candidates (wrong Beta({aw:.2f},{bw:.2f}), correct "
       f"Beta({ac:.2f},{bc:.2f}), AUROC {s['auroc_check']}): holds "
       f"{s['holds']}/{N_SEEDS}, leak mean {s['leak_mean']:.4f} "
       f"({s['leak_mean'] / ALPHA:.2f}xα), max {s['leak_max']:.4f} "
       f"({s['leak_max'] / ALPHA:.1f}xα)")
    md(f"- nominal KS rejection rates on exchangeable data: KS-1 "
       f"{s['ks1_reject']}/{N_SEEDS}, KS-2 {s['ks2_reject']}/{N_SEEDS} "
       f"(size validation: expected ~1/20 at p<0.05)")
    STATS["conformal"] = out


def part_perq():
    md("\n## 2. Per-question 部署语义指标（A3）\n")
    labels = load_labels()
    meta = load_pool_meta()
    b4, fuse, b1 = load_conf(B4_F), load_conf(FUSE_F), load_conf(B1_F)
    qids = sorted({k[0] for k in b4})
    out = {}
    for mname, conf in (("b4", b4), ("fuse_prod", fuse), ("b1", b1)):
        top = {}
        for q in qids:
            ks = [k for k in conf if k[0] == q]
            best = max(ks, key=lambda k: (conf[k], k))
            top[q] = (best, conf[best], labels[best] == "correct")
        pass1 = sum(1 for q in qids if top[q][2]) / len(qids)
        order = sorted(qids, key=lambda q: -top[q][1])
        sel = {}
        for cov in (0.3, 0.5, 0.7, 0.9):
            k = max(1, round(cov * len(order)))
            sel[cov] = sum(1 for q in order[:k] if top[q][2]) / k
        out[mname] = {"pass1": pass1, "selq": sel,
                      "top1_scores": {q: top[q][1] for q in qids}}
    single = {}
    for src in ("nucleus10", "beam10"):
        accs = [labels.get((q, src, 0)) == "correct" for q in qids]
        single[src + "_c0"] = sum(accs) / len(accs)
    oracle = sum(
        1 for q in qids
        if any(labels.get((q, s, i)) == "correct"
               for s in ("nucleus10", "beam10") for i in range(10)))
    out["single_sample"] = single
    out["oracle_top1"] = oracle / len(qids)
    out["n_questions"] = len(qids)
    order = sorted(qids, key=lambda q: -out["fuse_prod"]["top1_scores"][q])
    k50 = round(0.5 * len(order))
    comp = {}
    for half, qs in (("executed", order[:k50]), ("abstained", order[k50:])):
        comp[half] = dict(Counter(meta[(q, "nucleus10", 0)]["diff"] for q in qs))
    out["abstain_composition"] = comp
    md(f"- eval questions n={len(qids)}; oracle top-1 (pass@20) = "
       f"{out['oracle_top1']:.4f}")
    for m in ("b4", "fuse_prod", "b1"):
        md(f"- {m}: pass@1 (always execute top-1) = {out[m]['pass1']:.4f}; "
           f"per-question selective acc @30/50/70/90% questions = "
           f"{out[m]['selq'][0.3]:.4f}/{out[m]['selq'][0.5]:.4f}/"
           f"{out[m]['selq'][0.7]:.4f}/{out[m]['selq'][0.9]:.4f}")
    md(f"- single-sample baselines: nucleus10 first sample = "
       f"{single['nucleus10_c0']:.4f}, beam10 (T=0.2) first sample = "
       f"{single['beam10_c0']:.4f}")
    md(f"- abstention composition at 50% question coverage (fuse_prod): "
       f"{json.dumps(comp, ensure_ascii=False)}")
    STATS["perq"] = {k: v for k, v in out.items()
                     if k not in ("b4", "fuse_prod", "b1")}
    STATS["perq"]["methods"] = {
        m: {"pass1": out[m]["pass1"],
            "selq": {str(k): v for k, v in out[m]["selq"].items()}}
        for m in ("b4", "fuse_prod", "b1")}


def _fit_b4_seed(pool_data, seed, fit_frac=0.3):
    """Replicate run_b4_recalib fit/score for one seed. pool_data entries:
    (key, z, feats, correct, qid)."""
    qids = sorted({d[4] for d in pool_data})
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_fit = max(1, int(len(qids) * fit_frac))
    fit_q = set(qids[:n_fit])
    fit_idx = [i for i, d in enumerate(pool_data) if d[4] in fit_q]
    eva_idx = [i for i, d in enumerate(pool_data) if d[4] not in fit_q]
    dim = len(pool_data[0][2])
    mu = [sum(pool_data[i][2][j] for i in fit_idx) / len(fit_idx)
          for j in range(dim)]
    sd = [max(1e-6, math.sqrt(sum((pool_data[i][2][j] - mu[j]) ** 2
                                   for i in fit_idx) / len(fit_idx)))
          for j in range(dim)]

    def make_x(i):
        return ([(fj - m) / s for fj, m, s in zip(pool_data[i][2], mu, sd)]
                + [pool_data[i][1] / 10.0])

    params = fit_logistic([make_x(i) for i in fit_idx],
                          [pool_data[i][3] for i in fit_idx])

    def score(i):
        x = make_x(i)
        zz = params[-1] + sum(w * xj for w, xj in zip(params[:-1], x))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

    return {pool_data[i][0]: score(i) for i in eva_idx}


def part_seeds():
    md("\n## 3. 5 refit seeds 的 selEX 跨种子离散（B4 口径修正）\n")
    labels = load_labels()
    shares = {}
    for f in SH_F:
        shares.update(load_conf(f))
    pool = load_pool(POOL_F, require_logprobs=True)
    data = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue
        z = sum(c.token_logprobs) if c.token_logprobs else 0.0
        data.append((c.key, z, clause_features(c.sql), lab == "correct", c.qid))
    rows = {"b4": [], "fuse_prod": []}
    aur = {"b4": [], "fuse_prod": []}
    for seed in range(5):
        sc = _fit_b4_seed(data, seed)
        ks = sorted(sc)
        co_full = [labels[k] == "correct" for k in ks]
        cs = [(sc[k], labels[k] == "correct") for k in ks
              if labels[k] != "exec_fail"]
        aur["b4"].append(M.auroc([s for s, _ in cs], [c for _, c in cs]))
        rows["b4"].append(M.selective_ex([sc[k] for k in ks], co_full, 0.3))
        fsc = {k: sc[k] * shares.get(k, 0.5) for k in ks}
        fcs = [(fsc[k], labels[k] == "correct") for k in ks
               if labels[k] != "exec_fail"]
        aur["fuse_prod"].append(M.auroc([s for s, _ in fcs],
                                        [c for _, c in fcs]))
        rows["fuse_prod"].append(M.selective_ex([fsc[k] for k in ks],
                                                co_full, 0.3))
        print(f"[seeds] seed {seed} done", flush=True)

    def ms(v):
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        return m, sd

    out = {}
    for m in ("b4", "fuse_prod"):
        s30m, s30sd = ms(rows[m])
        am, asd = ms(aur[m])
        out[m] = {"selEX30_mean": s30m, "selEX30_std": s30sd,
                  "selEX30_values": rows[m], "auroc_mean": am,
                  "auroc_std": asd}
        md(f"- {m}: selEX@30 = {s30m:.4f} ± {s30sd:.4f} "
           f"(seeds {['%.4f' % x for x in rows[m]]}); "
           f"AUROC = {am:.4f} ± {asd:.4f}")
    STATS["seeds"] = out


def part_boot():
    md("\n## 4. Cluster bootstrap 口径统一（CI + 双侧 p）\n")
    labels = load_labels()
    b4, fuse, b1 = load_conf(B4_F), load_conf(FUSE_F), load_conf(B1_F)
    ks = sorted(b4)
    by_q = defaultdict(list)
    for k in ks:
        by_q[k[0]].append(k)
    qs = sorted(by_q)

    def pair_test(ca, cb, metric):
        if metric == "auroc":
            def f(cd, kk):
                return M.auroc([cd[k] for k in kk],
                               [labels[k] == "correct" for k in kk])
        else:
            # selEX@30 with the paper's deterministic tie-break (-conf, key)
            def f(cd, kk):
                ranked = sorted(kk, key=lambda k: (-cd[k], k))
                k30 = max(1, int(len(ranked) * 0.3))
                return sum(1 for k in ranked[:k30]
                           if labels[k] == "correct") / k30
        obs = f(cb, ks) - f(ca, ks)
        rng = random.Random(2026)
        deltas = []
        for _ in range(2000):
            qsamp = [qs[rng.randrange(len(qs))] for _ in range(len(qs))]
            kk = [k for q in qsamp for k in by_q[q]]
            deltas.append(f(cb, kk) - f(ca, kk))
        deltas.sort()
        lo = deltas[int(0.025 * len(deltas))]
        hi = deltas[int(0.975 * len(deltas)) - 1]
        p_le = sum(1 for d in deltas if d <= 0) / len(deltas)
        p_two = 2 * min(p_le, 1 - p_le)
        return {"delta": obs, "ci95": [lo, hi],
                "p_le0_onesided": p_le, "p_twosided": p_two}

    out = {}
    for tag, ca, cb, met in (
            ("fuse_vs_b1_auroc", b1, fuse, "auroc"),
            ("fuse_vs_b1_selex30", b1, fuse, "selex"),
            ("b4_vs_b1_auroc", b1, b4, "auroc"),
            ("b4_vs_b1_selex30", b1, b4, "selex"),
            ("fuse_vs_b4_auroc", b4, fuse, "auroc"),
            ("fuse_vs_b4_selex30", b4, fuse, "selex")):
        out[tag] = pair_test(ca, cb, met)
        r = out[tag]
        md(f"- {tag}: Δ={r['delta']:+.4f}, CI95 [{r['ci95'][0]:+.4f},"
           f"{r['ci95'][1]:+.4f}], p_le0(1-sided)={r['p_le0_onesided']:.4f}, "
           f"p_2sided={r['p_twosided']:.4f}")
    STATS["boot"] = out


def part_lodo():
    md("\n## 5. LODO 匹配对照（A4：同评估集 global-fit vs LODO）\n")
    labels = load_labels()
    shares = {}
    for f in SH_F:
        shares.update(load_conf(f))
    b4_eval = load_conf(B4_F)
    fuse_eval = load_conf(FUSE_F)
    pool = load_pool(POOL_F, require_logprobs=True)
    data = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue
        z = sum(c.token_logprobs) if c.token_logprobs else 0.0
        data.append((c, z, clause_features(c.sql), lab == "correct"))
    all_q = sorted({d[0].qid for d in data})
    rq = all_q[:]
    random.Random(0).shuffle(rq)
    fit_q = set(rq[:int(len(rq) * 0.3)])
    dbs = sorted({d[0].db_id for d in data})

    def logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    out = {"folds": []}
    for db in dbs:
        te = [i for i, d in enumerate(data) if d[0].db_id == db]
        tr = [i for i, d in enumerate(data) if d[0].db_id != db]
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
        l2_raw = [l2_x(oof[i], shares.get(data[i][0].key, 0.5))
                  for i in l2_fit]
        l2_y = [1.0 if data[i][3] else 0.0 for i in l2_fit]
        d2 = len(l2_raw[0])
        mu2 = [sum(r[j] for r in l2_raw) / len(l2_raw) for j in range(d2)]
        sd2 = [max(1e-6, math.sqrt(sum((r[j] - mu2[j]) ** 2 for r in l2_raw)
                                   / len(l2_raw))) for j in range(d2)]
        params_l2 = fit_logistic(
            [[(r[j] - mu2[j]) / sd2[j] for j in range(d2)] for r in l2_raw],
            l2_y)

        def fuse_of(i):
            x = l2_x(raw_score(i, params_l1),
                     shares.get(data[i][0].key, 0.5))
            zz = params_l2[-1] + sum(w * xj for w, xj in
                                     zip(params_l2[:-1],
                                         [(xj - m) / s for xj, m, s
                                          in zip(x, mu2, sd2)]))
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

        # matched subset: held-out DB candidates in the global eval set
        te_m = [i for i in te if data[i][0].qid not in fit_q
                and data[i][0].key in b4_eval]
        co_m = [data[i][3] for i in te_m]
        sc_lodo_b4 = [raw_score(i, params_l1) for i in te_m]
        sc_lodo_fu = [fuse_of(i) for i in te_m]
        sc_gb = [b4_eval[data[i][0].key] for i in te_m]
        sc_gf = [fuse_eval[data[i][0].key] for i in te_m]
        row = {
            "db": db, "n_matched": len(te_m),
            "b4_lodo": M.auroc(sc_lodo_b4, co_m),
            "b4_global": M.auroc(sc_gb, co_m),
            "fuse_lodo": M.auroc(sc_lodo_fu, co_m),
            "fuse_global": M.auroc(sc_gf, co_m),
            "b4_lodo_ece": M.ece(sc_lodo_b4, co_m),
            "b4_global_ece": M.ece(sc_gb, co_m),
            "fuse_lodo_ece": M.ece(sc_lodo_fu, co_m),
            "fuse_global_ece": M.ece(sc_gf, co_m),
        }
        out["folds"].append(row)
        md(f"- {db} (n_matched={row['n_matched']}): B4 "
           f"lodo {row['b4_lodo']:.3f} vs global {row['b4_global']:.3f} "
           f"(ECE {row['b4_lodo_ece']:.3f} vs {row['b4_global_ece']:.3f}); "
           f"fuse lodo {row['fuse_lodo']:.3f} vs global "
           f"{row['fuse_global']:.3f} (ECE {row['fuse_lodo_ece']:.3f} vs "
           f"{row['fuse_global_ece']:.3f})")
        print(f"[lodo] {db} done", flush=True)
    for m in ("b4", "fuse"):
        lodo_a = sum(f[f"{m}_lodo"] for f in out["folds"]) / len(out["folds"])
        glob_a = sum(f[f"{m}_global"] for f in out["folds"]) / len(out["folds"])
        lodo_e = sum(f[f"{m}_lodo_ece"] for f in out["folds"]) / len(out["folds"])
        glob_e = sum(f[f"{m}_global_ece"] for f in out["folds"]) / len(out["folds"])
        out[f"{m}_macro"] = {"lodo_auroc": lodo_a, "global_auroc": glob_a,
                             "d_auroc": lodo_a - glob_a,
                             "lodo_ece": lodo_e, "global_ece": glob_e,
                             "ece_rel_change": (lodo_e - glob_e) / glob_e}
        r = out[f"{m}_macro"]
        md(f"- MACRO {m}: ΔAUROC (lodo−global) = {r['d_auroc']:+.4f}; "
           f"ECE {r['global_ece']:.3f} → {r['lodo_ece']:.3f} "
           f"({r['ece_rel_change'] * 100:+.1f}%)")
    STATS["lodo"] = out


def part_misc():
    md("\n## 6. 杂项统计（B4 口径 / C 级数字）\n")
    from scipy import stats as st
    labels = load_labels()
    b4, fuse, iso = load_conf(B4_F), load_conf(FUSE_F), load_conf(ISO_F)
    ks = sorted(b4)
    out = {}
    th = {}
    for mname, conf in (("b4", b4), ("fuse_prod", fuse)):
        for t in (0.85, 0.9):
            sel = [k for k in ks if conf[k] >= t]
            n = len(sel)
            w = sum(1 for k in sel if labels[k] != "correct")
            if n:
                lo = st.beta.ppf(0.025, w, n - w + 1) if w > 0 else 0.0
                hi = st.beta.ppf(0.975, w + 1, n - w) if w < n else 1.0
                th[f"{mname}@{t}"] = {"n": n, "wrong": w, "rate": w / n,
                                      "cp95": [lo, hi]}
    out["thresholds"] = th
    for k, v in th.items():
        md(f"- purity {k}: n={v['n']}, wrong {v['wrong']}/{v['n']} "
           f"({v['rate'] * 100:.1f}%), CP95 [{v['cp95'][0]:.3f},"
           f"{v['cp95'][1]:.3f}]")
    # question-level bootstrap upper bound on leak (seed 0, B4)
    conf = b4
    qids = sorted({k[0] for k in ks})
    rq = qids[:]
    random.Random(0).shuffle(rq)
    cal_q = set(rq[:int(len(qids) * CAL_FRAC)])
    cal_w = sorted(1.0 - conf[k] for k in ks
                   if k[0] in cal_q and labels[k] != "correct")
    n = len(cal_w)
    test_by_q = {}
    for k in ks:
        if k[0] not in cal_q:
            test_by_q.setdefault(k[0], []).append(k)

    def leak_of(qq):
        aw = tw = 0
        for q in qq:
            for k in test_by_q[q]:
                if labels[k] != "correct":
                    tw += 1
                    if (1 + bisect.bisect_right(cal_w, 1.0 - conf[k])) \
                            / (n + 1) <= ALPHA:
                        aw += 1
        return aw / max(1, tw)

    tqs = sorted(test_by_q)
    obs_leak = leak_of(tqs)
    rng = random.Random(7)
    boots = sorted(leak_of([tqs[rng.randrange(len(tqs))]
                            for _ in range(len(tqs))]) for _ in range(2000))
    out["leak_boot_seed0_b4"] = {
        "obs": obs_leak, "ci95": [boots[int(0.025 * 2000)],
                                  boots[int(0.975 * 2000)]],
        "upper95": boots[int(0.95 * 2000)]}
    r = out["leak_boot_seed0_b4"]
    md(f"- leak (B4, seed 0) = {r['obs']:.4f}; question-level bootstrap 95% "
       f"CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}], one-sided upper 95% "
       f"{r['upper95']:.4f}")
    # isotonic tie-aware metrics
    iks = sorted(iso)
    co = [labels[k] == "correct" for k in iks]
    sc = [iso[k] for k in iks]
    groups = defaultdict(lambda: [0, 0])
    for s, c in zip(sc, co):
        groups[s][0 if c else 1] += 1
    vals = sorted(groups)
    conc = disc = 0
    for i, vi in enumerate(vals):
        ci, wi = groups[vi]  # ci = #correct, wi = #wrong at value vi
        # concordant: wrong below correct (correct scores higher)
        conc += wi * sum(groups[vj][0] for vj in vals[i + 1:])
        # discordant: correct below wrong
        disc += ci * sum(groups[vj][1] for vj in vals[i + 1:])
    n_all = len(sc)
    nc = sum(co)
    nw = n_all - nc
    n_pairs = nc * nw
    # Kendall tau-b on correct/wrong pairs: cw pairs tied in x are the
    # within-plateau ones; no cw pair is tied in y (labels differ by design).
    tied_x_cw = sum(x * y for x, y in groups.values())
    tau_b = (conc - disc) / math.sqrt(
        max(1e-9, (n_pairs - tied_x_cw) * n_pairs))
    out["iso"] = {
        "auroc_ties_half": M.auroc(sc, co),
        "n_plateaus": len(groups), "n_candidates": n_all,
        "strict_pairs": conc + disc, "strict_auroc": conc / max(1, conc + disc),
        "tied_cw_pairs": tied_x_cw, "cw_pairs": n_pairs,
        "kendall_tau_b": tau_b,
    }
    r = out["iso"]
    md(f"- isotonic B4: AUROC(ties=½) = {r['auroc_ties_half']:.3f}; "
       f"{r['n_plateaus']} plateaus over {r['n_candidates']} candidates; "
       f"strict-pair AUROC = {r['strict_auroc']:.3f} "
       f"({r['strict_pairs']}/{r['cw_pairs']} cw pairs non-tied); "
       f"Kendall τ-b (cw pairs) = {r['kendall_tau_b']:.3f}")
    if os.path.exists(DEV_F):
        dev = json.load(open(DEV_F, encoding="utf-8"))
        full = Counter(d["difficulty"] for d in dev)
        keep = set(json.load(open(os.path.join(POOLS, "auc300_qids.json"))))
        sub = Counter(dev[i]["difficulty"] for i in keep)
        out["composition"] = {"full": dict(full), "subset300": dict(sub),
                              "n_dev": len(dev)}
        md(f"- difficulty composition subset300 {dict(sub)} vs full dev "
           f"{dict(full)} (n_dev={len(dev)})")
    meta = load_pool_meta()
    uniq = {}
    for src in ("nucleus10", "beam10"):
        sqls = [v["sql"] for k, v in meta.items() if k[1] == src]
        norm = [" ".join(s.lower().split()) for s in sqls]
        uniq[src] = {"n": len(norm), "unique": len(set(norm))}
    both = [" ".join(v["sql"].lower().split()) for v in meta.values()]
    uniq["overall"] = {"n": len(both), "unique": len(set(both))}
    out["unique_sql"] = uniq
    md(f"- unique normalized SQL: nucleus10 {uniq['nucleus10']['unique']}/"
       f"{uniq['nucleus10']['n']}, beam10 {uniq['beam10']['unique']}/"
       f"{uniq['beam10']['n']}, overall {uniq['overall']['unique']}/"
       f"{uniq['overall']['n']}")
    STATS["misc"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    args = ap.parse_args()
    p = args.part
    if p in ("all", "conformal"):
        part_conformal()
    if p in ("all", "perq"):
        part_perq()
    if p in ("all", "seeds"):
        part_seeds()
    if p in ("all", "boot"):
        part_boot()
    if p in ("all", "lodo"):
        part_lodo()
    if p in ("all", "misc"):
        part_misc()
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    body = "\n".join(MD) + "\n"
    if p != "all" and os.path.exists(OUT_MD):
        prev = open(OUT_MD, encoding="utf-8").read()
        if prev:
            body = prev.rstrip("\n") + "\n\n" + body.split("\n", 1)[1].lstrip("\n")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(body)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    merged = {}
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, encoding="utf-8") as f:
                merged.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    merged.update(STATS)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=1, ensure_ascii=False, default=str)
    print(f"[done] md -> {OUT_MD}\n      json -> {OUT_JSON}")


if __name__ == "__main__":
    main()
