# -*- coding: utf-8 -*-
"""p1_analysis.py — P1 revision experiments (meta-review round 1, P1 items).

Sections (run: python p1_analysis.py --part all|execache|cfix|lodoref|ksens|
                                          select|b2mech|b3stats|b4stats|b1disc|
                                          driftshare|figs):
  execache  : execute every unique (db, sql) in the pool + gold queries; cache
              strict/lenient result-set hashes, row counts, error classes.
              (Prerequisite for select / b2mech / b3stats / driftshare.)
  cfix      : A1.4 cluster-aware conformal repairs — question-unit (top-1)
              conformal, per-question subsampled calibration (Dunn et al.),
              random-unit question-marginal conformal, at alpha in
              {0.02, 0.05, 0.10}, 20 seeds, B4 + fuse_prod.
  lodoref   : A5.1 target-DB few-shot refit — per LODO fold, refit B4 with
              k in {5,10,20} labeled target-DB questions (target-only /
              global+k / intercept-only) vs frozen global; AUROC/ECE over
              5 draws.
  ksens     : A3.4 K-sensitivity — share (B2) and product-fusion quality
              under K in {5,10,20} voting-pool sizes.
  select    : A3.2 per-question selection baselines (execution-majority,
              B5-as-selector, B4/fuse top-1, single-sample, random, oracle)
              + A3.3 two/three-action cost model and optimal operating
              points on the per-question risk-coverage curve.
  b2mech    : B2(d) plurality-error-vs-share conditional curve (true
              result-set plurality from the execution cache).
  b3stats   : B3 partial — exec_fail error taxonomy (timeout vs error),
              LIMIT-without-ORDER-BY audit (5x re-execution stability),
              lenient-match upper bound on silent-label noise.
  b4stats   : B4 remaining — ECE10/ECE15/Brier with question-level cluster
              bootstrap CIs; high-end purity at aligned 0.85/0.9 thresholds
              with cluster-bootstrap CIs.
  b1disc    : B1 appendix stats — B1/B5 score distributions, unique-SQL
              counts, Protocol B intersection count.
  driftshare: B3 — share (B2) recomputed on drifted databases (rename/
              add_col/split_tbl): does consistency voting survive drift?
  figs      : C13 pipeline overview, C14 risk-coverage 5-seed uncertainty
              band, B1/B5 histograms, K-sensitivity curve.

Outputs: results/revision_p1.md + paper/results/_p1_stats.json
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for p0 import

from common.schema import load_pool  # noqa: E402
from common import metrics as M  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402
from p0_reanalysis import (ALPHA, B1_F, B4_F, FUSE_F, LAB_F, POOL_F, SH_F,  # noqa: E402
                           CAL_FRAC, N_SEEDS, load_conf, load_labels,
                           load_pool_meta)

OUT_MD = os.path.join(_ROOT, "results", "revision_p1.md")
OUT_JSON = os.path.join(_ROOT, "results", "_p1_stats.json")
CACHE_F = os.path.join(_ROOT, "data", "pools", "p1_result_cache.jsonl")
FIG_DIR = os.path.join(_ROOT, "figures")

STATS: dict = {}
MD: list[str] = ["# P1 补充实验（revision round 1, P1）\n"]


def md(s: str = "") -> None:
    MD.append(s)


def _mean_sd(v):
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return m, sd


# ====================================================================== #
# execache                                                              #
# ====================================================================== #
def run_sql_cls(db_path: str, sql: str, timeout: int = 10, watchdog: float = 30.0):
    """Like exec_match.run_sql but classifies failures. Returns
    (result_set_or_None, err_class) with err_class in
    {None, 'timeout', 'locked', 'error', 'no_db'}."""
    if not os.path.exists(db_path):
        return None, "no_db"
    conn = None
    timer = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        if watchdog:
            timer = threading.Timer(watchdog, conn.interrupt)
            timer.daemon = True
            timer.start()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return sorted(tuple(r) for r in rows), None
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "interrupted" in msg:
            return None, "timeout"
        if "locked" in msg or "busy" in msg:
            return None, "locked"
        return None, "error"
    finally:
        if timer:
            timer.cancel()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _sha(x) -> str:
    return hashlib.sha1(repr(x).encode("utf-8", "replace")).hexdigest()[:16]


def _lenient_val(v):
    if v is None:
        return ("z",)
    if isinstance(v, bool):
        return ("n", float(v))
    if isinstance(v, (int, float)):
        return ("n", round(float(v), 4))
    if isinstance(v, bytes):
        return ("b", _sha(v))
    return ("s", str(v).strip().lower())


def _result_hashes(rows):
    strict = _sha(rows)
    lenient = _sha(sorted(tuple(_lenient_val(v) for v in r) for r in rows))
    return strict, lenient


def _db_path(db_id: str) -> str:
    from common.exec_match import db_path_of
    return db_path_of(db_id, "bird")


def part_execache():
    md("\n## 0. 执行结果缓存（unique SQL 严格/宽松哈希 + 错误分类）\n")
    pool = load_pool(POOL_F)
    done: dict[str, dict] = {}
    if os.path.exists(CACHE_F):
        with open(CACHE_F, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["db"] + "\x00" + r["sql"]] = r
    tasks: dict[str, tuple[str, str]] = {}
    gold_sqls: dict[int, tuple[str, str]] = {}  # qid -> (db, gold)
    for c in pool:
        # raw (stripped) SQL: _norm_sql lowercases string literals, which
        # changes semantics under SQLite's case-sensitive BINARY collation
        # (the original labeling executed raw SQL)
        ns = c.sql.strip()
        tasks.setdefault(c.db_id + "\x00" + ns, (c.db_id, ns))
        if c.qid not in gold_sqls:
            gold_sqls[c.qid] = (c.db_id, c.gold_sql.strip())
    for qid, (db, g) in gold_sqls.items():
        tasks.setdefault(db + "\x00" + g, (db, g))
    todo = [(k, v) for k, v in tasks.items() if k not in done]
    print(f"[execache] unique SQL: {len(tasks)}, to execute: {len(todo)}",
          flush=True)

    def work(item):
        key, (db, ns) = item
        res, err = run_sql_cls(_db_path(db), ns)
        if res is None:
            return key, {"db": db, "sql": ns, "ok": False, "err": err,
                         "strict_h": None, "len_h": None, "n_rows": None}
        sh, lh = _result_hashes(res)
        return key, {"db": db, "sql": ns, "ok": True, "err": None,
                     "strict_h": sh, "len_h": lh, "n_rows": len(res)}

    n_done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for key, rec in ex.map(work, todo):
                done[key] = rec
                n_done += 1
                if n_done % 200 == 0:
                    print(f"[execache] {n_done}/{len(todo)}", flush=True)
                    with open(CACHE_F, "w", encoding="utf-8") as f:
                        for r in done.values():
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(CACHE_F, "w", encoding="utf-8") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # sanity: label consistency (correct <=> strict_h == gold strict_h)
    labels = load_labels()
    gold_h = {}
    for c in pool:
        if c.qid not in gold_h:
            g = c.gold_sql.strip()
            rec = done.get(c.db_id + "\x00" + g)
            gold_h[c.qid] = rec["strict_h"] if rec and rec["ok"] else None
    agree = disagree = 0
    cand_h = {}
    for c in pool:
        ns = c.sql.strip()
        rec = done.get(c.db_id + "\x00" + ns)
        cand_h[c.key] = rec["strict_h"] if rec and rec["ok"] else None
    for k, lab in labels.items():
        ch, gh = cand_h.get(k), gold_h.get(k[0])
        if ch is None or gh is None:
            continue
        if (ch == gh) == (lab == "correct"):
            agree += 1
        else:
            disagree += 1
    out = {"n_unique": len(tasks), "n_ok": sum(1 for r in done.values() if r["ok"]),
           "label_agree": agree, "label_disagree": disagree}
    err_types = Counter(r["err"] for r in done.values() if not r["ok"])
    out["err_types"] = dict(err_types)
    md(f"- unique SQL {out['n_unique']} ({out['n_ok']} executed ok); "
       f"failures: {dict(err_types)}")
    md(f"- label consistency vs cache: agree {agree}, disagree {disagree}")
    STATS["execache"] = out


def load_cache() -> dict:
    d = {}
    with open(CACHE_F, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[r["db"] + "\x00" + r["sql"]] = r
    return d


def cand_hashes() -> tuple[dict, dict]:
    """(cand_key -> strict_h or None, qid -> gold strict_h or None)."""
    cache = load_cache()
    pool = load_pool(POOL_F)
    ch, gh = {}, {}
    for c in pool:
        ns = c.sql.strip()
        rec = cache.get(c.db_id + "\x00" + ns)
        ch[c.key] = rec["strict_h"] if rec and rec["ok"] else None
        if c.qid not in gh:
            g = c.gold_sql.strip()
            grec = cache.get(c.db_id + "\x00" + g)
            gh[c.qid] = grec["strict_h"] if grec and grec["ok"] else None
    return ch, gh


def cand_lenient() -> tuple[dict, dict]:
    cache = load_cache()
    pool = load_pool(POOL_F)
    cl, gl = {}, {}
    for c in pool:
        ns = c.sql.strip()
        rec = cache.get(c.db_id + "\x00" + ns)
        cl[c.key] = rec["len_h"] if rec and rec["ok"] else None
        if c.qid not in gl:
            g = c.gold_sql.strip()
            grec = cache.get(c.db_id + "\x00" + g)
            gl[c.qid] = grec["len_h"] if grec and grec["ok"] else None
    return cl, gl


# ====================================================================== #
# cfix (A1.4)                                                           #
# ====================================================================== #
def _pval(cal_w, x):
    return (1 + bisect.bisect_right(cal_w, x)) / (len(cal_w) + 1)


def _eval_units(cal_w, units, alpha):
    """units: list of (nonconf, is_wrong). Returns leak/holds/coverage."""
    if not cal_w or not units:
        return None
    n_acc = n_wrong = acc_wrong = 0
    for x, wrong in units:
        if _pval(cal_w, x) <= alpha:
            n_acc += 1
            if wrong:
                acc_wrong += 1
        if wrong:
            n_wrong += 1
    if n_wrong == 0:
        return None
    leak = acc_wrong / n_wrong
    return {"leak": leak, "holds": leak <= alpha + 1e-9,
            "coverage": n_acc / len(units), "n_cal_wrong": len(cal_w),
            "n_test_wrong": n_wrong}


def part_cfix():
    md("\n## 1. A1.4 簇感知共形修复（B4 / fuse_prod, 20 seeds）\n")
    labels = load_labels()
    out = {}
    for mname, path in (("b4", B4_F), ("fuse_prod", FUSE_F)):
        conf = load_conf(path)
        keys = sorted(conf)
        by_q = defaultdict(list)
        for k in keys:
            by_q[k[0]].append(k)
        qids = sorted(by_q)
        n_cal = int(len(qids) * CAL_FRAC)
        for alpha in (0.02, 0.05, 0.10):
            rows = {"qtop1": [], "subcal": [], "qrand": []}
            for seed in range(N_SEEDS):
                rq = qids[:]
                random.Random(seed).shuffle(rq)
                cal_q = set(rq[:n_cal])
                test_q = [q for q in qids if q not in cal_q]
                test_set = set(test_q)
                rng = random.Random(100_000 + seed)
                # V1: deployment unit = top-1 candidate per question
                top1 = {q: max(by_q[q], key=lambda k: (conf[k], k))
                        for q in qids}
                cal_w1 = sorted(1.0 - conf[top1[q]] for q in cal_q
                                if labels[top1[q]] != "correct")
                u1 = [(1.0 - conf[top1[q]], labels[top1[q]] != "correct")
                      for q in test_q]
                r1 = _eval_units(cal_w1, u1, alpha)
                if r1:
                    rows["qtop1"].append(r1)
                # V2: subsampled calibration — one random wrong candidate per
                #     cal question (Dunn et al. subsampling); candidate-level
                #     test (same quantity as the original B8 audit)
                cal_w2 = []
                for q in cal_q:
                    wrongs = [k for k in by_q[q] if labels[k] != "correct"]
                    if wrongs:
                        cal_w2.append(1.0 - conf[rng.choice(wrongs)])
                cal_w2.sort()
                u2 = [(1.0 - conf[k], labels[k] != "correct") for k in keys
                      if k[0] in test_set]
                r2 = _eval_units(cal_w2, u2, alpha)
                if r2:
                    rows["subcal"].append(r2)
                # V3: question-marginal — one random candidate per question on
                #     both sides (cal keeps only wrong units, as B8 does)
                cal_w3 = []
                for q in cal_q:
                    k = rng.choice(by_q[q])
                    if labels[k] != "correct":
                        cal_w3.append(1.0 - conf[k])
                cal_w3.sort()
                u3 = []
                for q in test_q:
                    k = rng.choice(by_q[q])
                    u3.append((1.0 - conf[k], labels[k] != "correct"))
                r3 = _eval_units(cal_w3, u3, alpha)
                if r3:
                    rows["qrand"].append(r3)
            for vname, rr in rows.items():
                if not rr:
                    continue
                leaks = [r["leak"] for r in rr]
                covs = [r["coverage"] for r in rr]
                ncw = [r["n_cal_wrong"] for r in rr]
                key = f"{mname}_a{alpha}_{vname}"
                out[key] = {
                    "alpha": alpha, "holds": sum(r["holds"] for r in rr),
                    "n_seeds": len(rr), "leak_mean": sum(leaks) / len(leaks),
                    "leak_median": sorted(leaks)[len(leaks) // 2],
                    "leak_max": max(leaks), "coverage_mean": sum(covs) / len(covs),
                    "n_cal_wrong_mean": sum(ncw) / len(ncw),
                    "vacuous_seeds": sum(1 for r in rr if r["coverage"] == 0.0),
                }
                r = out[key]
                md(f"- {mname} α={alpha} [{vname}]: holds {r['holds']}/"
                   f"{r['n_seeds']} (vacuous {r['vacuous_seeds']}), leak mean "
                   f"{r['leak_mean']:.4f} ({r['leak_mean'] / alpha:.2f}×α), "
                   f"median {r['leak_median']:.4f}, max {r['leak_max']:.4f} "
                   f"({r['leak_max'] / alpha:.1f}×α), coverage "
                   f"{r['coverage_mean']:.3f}, cal-wrong n≈{r['n_cal_wrong_mean']:.0f}")
    STATS["cfix"] = out


# ====================================================================== #
# lodoref (A5.1)                                                        #
# ====================================================================== #
def _fit_b4(train_idx, data):
    dim = len(data[0][2])
    mu = [sum(data[i][2][j] for i in train_idx) / len(train_idx)
          for j in range(dim)]
    sd = [max(1e-6, math.sqrt(sum((data[i][2][j] - mu[j]) ** 2
                                   for i in train_idx) / len(train_idx)))
          for j in range(dim)]

    def make_x(i):
        return ([(fj - m) / s for fj, m, s in zip(data[i][2], mu, sd)]
                + [data[i][1] / 10.0])

    params = fit_logistic([make_x(i) for i in train_idx],
                          [data[i][3] for i in train_idx])

    def score(i):
        x = make_x(i)
        zz = params[-1] + sum(w * xj for w, xj in zip(params[:-1], x))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

    return score, params, mu, sd, make_x


def _fit_intercept_only(train_idx, data, params, make_x):
    """Platt-style: global weights, refit only the intercept on target data."""
    b = params[-1]
    w = params[:-1]
    n = len(train_idx)
    for _ in range(300):
        gb = 0.0
        for i in train_idx:
            x = make_x(i)
            zz = b + sum(wj * xj for wj, xj in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))
            gb += p - (1.0 if data[i][3] else 0.0)
        b -= 0.5 * gb / n
    return b


def part_lodoref():
    md("\n## 2. A5.1 目标库小样本重拟合（B4 校准层, 11 folds × k∈{5,10,20}）\n")
    labels = load_labels()
    b4_eval = load_conf(B4_F)
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
    out = {"folds": [], "n_dbs": len(dbs)}
    KS = (5, 10, 20)
    DRAWS = 5
    for db in dbs:
        te_global = [i for i, d in enumerate(data)
                     if d[0].db_id == db and d[0].qid not in fit_q
                     and d[0].key in b4_eval]
        te_pool_q = sorted({data[i][0].qid for i in te_global})
        tr = [i for i, d in enumerate(data) if d[0].db_id != db]
        score_g, params_g, mu_g, sd_g, make_x_g = _fit_b4(tr, data)
        row = {"db": db, "n_eval_pool_q": len(te_pool_q),
               "n_eval_cand": len(te_global), "ks": {}}
        co_g = [data[i][3] for i in te_global]
        row["frozen"] = {"auroc": M.auroc([score_g(i) for i in te_global], co_g),
                         "ece": M.ece([score_g(i) for i in te_global], co_g)}
        for k in KS:
            if len(te_pool_q) < k + 5:
                row["ks"][k] = None
                continue
            agg = {v: {"auroc": [], "ece": []}
                   for v in ("target_only", "global_plus_k", "intercept")}
            for draw in range(DRAWS):
                rng = random.Random(7_000 + 31 * draw + k)
                chosen = set(rng.sample(te_pool_q, k))
                ev = [i for i in te_global if data[i][0].qid not in chosen]
                tr_k = [i for i, d in enumerate(data) if d[0].qid in chosen]
                co = [data[i][3] for i in ev]
                if not any(co) or all(co):
                    continue
                s_t, _, _, _, _ = _fit_b4(tr_k, data)
                agg["target_only"]["auroc"].append(
                    M.auroc([s_t(i) for i in ev], co))
                agg["target_only"]["ece"].append(M.ece([s_t(i) for i in ev], co))
                s_gk, _, _, _, _ = _fit_b4(tr + tr_k, data)
                agg["global_plus_k"]["auroc"].append(
                    M.auroc([s_gk(i) for i in ev], co))
                agg["global_plus_k"]["ece"].append(M.ece([s_gk(i) for i in ev], co))
                b_new = _fit_intercept_only(tr_k, data, params_g, make_x_g)

                def s_int(i, b=b_new):
                    x = make_x_g(i)
                    zz = b + sum(w * xj for w, xj in zip(params_g[:-1], x))
                    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

                agg["intercept"]["auroc"].append(M.auroc([s_int(i) for i in ev], co))
                agg["intercept"]["ece"].append(M.ece([s_int(i) for i in ev], co))
            row["ks"][k] = {}
            for v, d in agg.items():
                if not d["auroc"]:
                    row["ks"][k][v] = None
                    continue
                am, asd = _mean_sd(d["auroc"])
                em, esd = _mean_sd(d["ece"])
                row["ks"][k][v] = {"auroc_mean": am, "auroc_sd": asd,
                                   "ece_mean": em, "ece_sd": esd,
                                   "n_draws": len(d["auroc"])}
        out["folds"].append(row)
        print(f"[lodoref] {db} done", flush=True)
    # aggregate macro stats
    agg = {}
    for k in KS:
        agg[k] = {}
        for v in ("frozen", "target_only", "global_plus_k", "intercept"):
            if v == "frozen":
                vals_a = [f["frozen"]["auroc"] for f in out["folds"]]
                vals_e = [f["frozen"]["ece"] for f in out["folds"]]
                am, asd = _mean_sd(vals_a)
                em, esd = _mean_sd(vals_e)
                agg[k][v] = {"auroc_mean": am, "auroc_sd": asd,
                             "ece_mean": em, "ece_sd": esd}
            else:
                rows = [f["ks"][k][v] for f in out["folds"]
                        if f["ks"].get(k) and f["ks"][k].get(v)]
                if not rows:
                    agg[k][v] = None
                    continue
                am, asd = _mean_sd([r["auroc_mean"] for r in rows])
                em, esd = _mean_sd([r["ece_mean"] for r in rows])
                pool_sd = math.sqrt(sum(r["auroc_sd"] ** 2 for r in rows)
                                    / len(rows))
                e_pool_sd = math.sqrt(sum(r["ece_sd"] ** 2 for r in rows)
                                      / len(rows))
                agg[k][v] = {"auroc_mean": am, "auroc_macro_sd": asd,
                             "auroc_within_sd": pool_sd,
                             "ece_mean": em, "ece_macro_sd": esd,
                             "ece_within_sd": e_pool_sd, "n_folds": len(rows)}
    out["macro"] = {str(k): v for k, v in agg.items()}
    for k in KS:
        md(f"### k={k}\n")
        a = agg[k]
        f = a["frozen"]
        md(f"- frozen global: AUROC {f['auroc_mean']:.3f}±{f['auroc_sd']:.3f} "
           f"(macro over folds), ECE {f['ece_mean']:.3f}±{f['ece_sd']:.3f}")
        for v in ("target_only", "global_plus_k", "intercept"):
            if not a[v]:
                md(f"- {v}: infeasible (no fold with eval pool ≥ k+5)")
                continue
            r = a[v]
            md(f"- {v}: AUROC {r['auroc_mean']:.3f} "
               f"(macro sd {r['auroc_macro_sd']:.3f}, within-fold sd "
               f"{r['auroc_within_sd']:.3f}), ECE {r['ece_mean']:.3f} "
               f"(within {r['ece_within_sd']:.3f}), folds {r['n_folds']}")
    n_k20 = sum(1 for f in out["folds"] if f["ks"].get(20))
    md(f"- k=20 feasible on {n_k20}/{len(dbs)} folds "
       f"(requires ≥25 eval questions in the target DB)")
    STATS["lodoref"] = out


# ====================================================================== #
# ksens (A3.4)                                                          #
# ====================================================================== #
def part_ksens():
    md("\n## 3. A3.4 K 敏感性（share 信号与乘积融合, seed-0 切分）\n")
    from p0_reanalysis import _fit_b4_seed
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
        # truncated z, exactly as run_b4_recalib (raw sum differs on 2
        # extreme candidates and breaks file reproduction)
        z = math.log(max(math.exp(sum(c.token_logprobs)
                                  if c.token_logprobs else 0.0), 1e-12))
        data.append((c.key, z, clause_features(c.sql), lab == "correct",
                     c.qid, c.source))
    sc = _fit_b4_seed([(d[0], d[1], d[2], d[3], d[4]) for d in data], 0)
    ks = sorted(sc)
    groups = defaultdict(list)
    for d in data:
        groups[(d[4], d[5])].append(d[0])
    out = {}

    def share_of_group(member_keys, subset=None):
        valid = [labels[k] for k in (subset or member_keys)
                 if labels[k] in ("correct", "silent")]
        if not valid:
            return 0.0
        return max(valid.count("correct"), valid.count("silent")) / len(valid)

    def eval_with(share_map):
        # paper accounting: full eval set, exec_fail counted as wrong
        fsc = {k: sc[k] * share_map.get(k, 0.5) for k in ks}
        co = [labels[k] == "correct" for k in ks]
        return {
            "b2_auroc": M.auroc([share_map.get(k, 0.5) for k in ks], co),
            "fuse_auroc": M.auroc([fsc[k] for k in ks], co),
            "fuse_ece": M.ece([fsc[k] for k in ks], co),
            "fuse_selex30": M.selective_ex([fsc[k] for k in ks], co, 0.3),
        }

    # K = 10 (original per-group share)
    out["K10"] = eval_with(shares)
    # K = 5 (subsample 5 of each group, 5 draws)
    draws = []
    for d in range(5):
        rng = random.Random(50_000 + d)
        sm = {}
        for g, mks in groups.items():
            sub = rng.sample(mks, 5)
            s = share_of_group(mks, sub)
            for k in mks:
                sm[k] = s
        draws.append(eval_with(sm))
    out["K5"] = {m: {"mean": _mean_sd([r[m] for r in draws])[0],
                     "sd": _mean_sd([r[m] for r in draws])[1]}
                 for m in draws[0]}
    # K = 20 (pooled per-question vote)
    by_q = defaultdict(list)
    for d in data:
        by_q[d[4]].append(d[0])
    sm20 = {}
    for q, qks in by_q.items():
        s = share_of_group(qks)
        for k in qks:
            sm20[k] = s
    out["K20_pooled"] = eval_with(sm20)
    for name in ("K5", "K10", "K20_pooled"):
        r = out[name]
        if name == "K5":
            md(f"- {name}: B2 AUROC {r['b2_auroc']['mean']:.4f}±"
               f"{r['b2_auroc']['sd']:.4f}, fuse AUROC "
               f"{r['fuse_auroc']['mean']:.4f}±{r['fuse_auroc']['sd']:.4f}, "
               f"fuse ECE {r['fuse_ece']['mean']:.4f}±"
               f"{r['fuse_ece']['sd']:.4f}, selEX@30 "
               f"{r['fuse_selex30']['mean']:.4f}±{r['fuse_selex30']['sd']:.4f}")
        else:
            md(f"- {name}: B2 AUROC {r['b2_auroc']:.4f}, fuse AUROC "
               f"{r['fuse_auroc']:.4f}, fuse ECE {r['fuse_ece']:.4f}, "
               f"selEX@30 {r['fuse_selex30']:.4f}")
    b4f = load_conf(B4_F)
    fpf = load_conf(FUSE_F)
    d_b4 = max(abs(sc[k] - b4f[k]) for k in ks)
    d_fp = max(abs(sc[k] * shares.get(k, 0.5) - fpf[k]) for k in ks)
    md(f"- K=10 sanity: recomputation vs files max|Δb4|={d_b4:.1e}, "
       f"max|Δfuse|={d_fp:.1e}; fuse AUROC {out['K10']['fuse_auroc']:.4f} "
       f"(paper seed-0: 0.6655), ECE {out['K10']['fuse_ece']:.4f} (0.1205), "
       f"selEX@30 {out['K10']['fuse_selex30']:.4f} (0.708)")
    out["sanity"] = {"max_d_b4": d_b4, "max_d_fuse": d_fp}
    STATS["ksens"] = out


# ====================================================================== #
# select (A3.2 + A3.3)                                                  #
# ====================================================================== #
def part_select():
    md("\n## 4. A3.2 同池 selection 基线 + A3.3 三动作代价模型（eval 题 n=210）\n")
    labels = load_labels()
    meta = load_pool_meta()
    b4, fuse, b1 = load_conf(B4_F), load_conf(FUSE_F), load_conf(B1_F)
    b5 = load_conf(os.path.join(os.path.dirname(B1_F), "bird300_ds_b5.jsonl"))
    qids = sorted({k[0] for k in b4})
    ch, gh = cand_hashes()

    def top1_by(conf, q):
        ks = [k for k in conf if k[0] == q]
        return max(ks, key=lambda k: (conf[k], k))

    # execution-majority (plurality over result sets of all 20 candidates)
    def exec_majority(q):
        classes = defaultdict(list)
        for src in ("nucleus10", "beam10"):
            for i in range(10):
                k = (q, src, i)
                h = ch.get(k)
                if h is not None:
                    classes[h].append(k)
        if not classes:
            return False
        best_cls = min(classes.items(),
                       key=lambda kv: (-len(kv[1]), min(kv[1])))
        return best_cls[0] == gh.get(q)

    rand_accs = []
    for d in range(5):
        rng = random.Random(60_000 + d)
        acc = 0
        for q in qids:
            src = rng.choice(("nucleus10", "beam10"))
            i = rng.randrange(10)
            acc += labels.get((q, src, i)) == "correct"
        rand_accs.append(acc / len(qids))
    em = {q: exec_majority(q) for q in qids}
    b5t = {q: labels[top1_by(b5, q)] == "correct" for q in qids}
    b4t = {q: labels[top1_by(b4, q)] == "correct" for q in qids}
    fut = {q: labels[top1_by(fuse, q)] == "correct" for q in qids}
    p1 = {q: labels.get((q, "nucleus10", 0)) == "correct" for q in qids}
    orc = {q: any(labels.get((q, s, i)) == "correct"
                  for s in ("nucleus10", "beam10") for i in range(10))
           for q in qids}
    scores_fuse = {q: fuse[top1_by(fuse, q)] for q in qids}

    def acc(d):
        return sum(1 for q in qids if d[q]) / len(qids)

    def prop_ci(x, n):
        from scipy import stats as st
        if x == 0:
            return [0.0, st.beta.ppf(0.975, 1, n)]
        if x == n:
            return [st.beta.ppf(0.025, n, 1), 1.0]
        return [st.beta.ppf(0.025, x, n - x + 1),
                st.beta.ppf(0.975, x + 1, n - x)]

    methods = {
        "random_1": rand_accs,
        "pass@1_nucleus_c0": p1,
        "exec_majority": em,
        "b5_top1": b5t,
        "b4_top1": b4t,
        "fuse_top1": fut,
        "oracle_top1": orc,
    }
    out = {"n_questions": len(qids)}
    for name, d in methods.items():
        if isinstance(d, list):
            m, sd = _mean_sd(d)
            x = round(m * len(qids))
            ci = prop_ci(x, len(qids))
            out[name] = {"acc": m, "sd": sd, "cp95": ci}
            md(f"- {name}: {m:.4f} ± {sd:.4f} (CP95 [{ci[0]:.3f},"
               f"{ci[1]:.3f}])")
        else:
            a = acc(d)
            x = sum(1 for q in qids if d[q])
            ci = prop_ci(x, len(qids))
            out[name] = {"acc": a, "cp95": ci}
            md(f"- {name}: {a:.4f} (CP95 [{ci[0]:.3f},{ci[1]:.3f}])")
    # paired question bootstrap: fuse vs exec_majority / b5
    qs = qids[:]

    def pair_da(da, db_):
        rng = random.Random(2027)
        deltas = []
        for _ in range(2000):
            samp = [qs[rng.randrange(len(qs))] for _ in range(len(qs))]
            deltas.append(sum(1 for q in samp if db_[q]) / len(samp)
                          - sum(1 for q in samp if da[q]) / len(samp))
        deltas.sort()
        return {"delta": acc(db_) - acc(da),
                "ci95": [deltas[50], deltas[1949]],
                "p_le0": sum(1 for x in deltas if x <= 0) / len(deltas)}

    for tag, da, db_ in (("fuse_vs_execmaj", em, fut),
                         ("fuse_vs_b5", b5t, fut)):
        r = pair_da(da, db_)
        out[tag] = r
        md(f"- {tag}: Δ={r['delta']:+.4f}, CI95 [{r['ci95'][0]:+.4f},"
           f"{r['ci95'][1]:+.4f}], p(Δ≤0)={r['p_le0']:.4f}")

    # ---- A3.3 cost model ----
    md("### A3.3 代价模型（fuse top-1 分数, 错误执行代价=1）\n")
    wrong = {q: not fut[q] for q in qids}
    binary = {}
    for a in (0.05, 0.10, 0.20, 0.30, 0.50):
        tau = 1.0 - a
        ex = [q for q in qids if scores_fuse[q] >= tau]
        cost = (sum(1 for q in ex if wrong[q]) + a * (len(qids) - len(ex))) \
            / len(qids)
        binary[a] = {"tau": tau, "coverage": len(ex) / len(qids),
                     "avg_cost": cost,
                     "exec_risk": (sum(1 for q in ex if wrong[q]) / len(ex))
                     if ex else None}
        md(f"- execute/abstain a={a:.2f}: τ={tau:.2f}, coverage "
           f"{len(ex)}/{len(qids)} ({len(ex) / len(qids):.2f}), "
           f"avg cost {cost:.4f} (always-execute {sum(wrong.values()) / len(qids):.4f})")
    three = {}
    for a in (0.05, 0.10, 0.20):
        for d_ in (0.30, 0.50, 0.70):
            if d_ <= a:
                continue
            n_ex = n_de = n_ab = 0
            cost = 0.0
            for q in qids:
                s = scores_fuse[q]
                if 1.0 - s <= a:
                    n_ex += 1
                    cost += 1.0 if wrong[q] else 0.0
                elif 1.0 - s <= d_:
                    n_de += 1
                    cost += d_
                else:
                    n_ab += 1
                    cost += a
            three[f"a{a}_d{d_}"] = {
                "exec": n_ex, "defer": n_de, "abstain": n_ab,
                "avg_cost": cost / len(qids)}
            md(f"- 3-action a={a:.2f}, d={d_:.2f}: exec/defer/abstain = "
               f"{n_ex}/{n_de}/{n_ab}, avg cost {cost / len(qids):.4f}")
    out["cost_binary"] = {str(k): v for k, v in binary.items()}
    out["cost_three"] = three
    out["fuse_top1_scores"] = scores_fuse
    out["wrong"] = wrong
    STATS["select"] = {k: v for k, v in out.items()
                       if k not in ("fuse_top1_scores", "wrong")}


# ====================================================================== #
# b2mech (B2(d))                                                        #
# ====================================================================== #
def part_b2mech():
    md("\n## 5. B2(d) plurality 错误率随 share 的条件曲线（全部 600 组）\n")
    labels = load_labels()
    meta = load_pool_meta()
    ch, gh = cand_hashes()
    shares = {}
    for f in SH_F:
        shares.update(load_conf(f))
    groups = defaultdict(list)
    for k in labels:
        groups[(k[0], k[1])].append(k)
    buckets = defaultdict(lambda: [0, 0, 0.0])  # n, n_wrong_plur, sum true_share
    tot = {"n": 0, "wrong": 0}
    for g, mks in sorted(groups.items()):
        valid = [k for k in mks if labels[k] in ("correct", "silent")]
        if not valid:
            continue
        classes = defaultdict(int)
        for k in valid:
            h = ch.get(k)
            if h is not None:
                classes[h] += 1
        if not classes:
            continue
        if gh.get(g[0]) is None:  # gold re-execution failed: cannot judge
            tot["skipped"] = tot.get("skipped", 0) + 1
            continue
        plur, size = max(classes.items(), key=lambda kv: (kv[1], kv[0]))
        plur_wrong = plur != gh.get(g[0])
        true_share = size / len(valid)
        label_share = shares.get(mks[0], 0.0)
        b = round(label_share, 1)
        buckets[b][0] += 1
        buckets[b][1] += 1 if plur_wrong else 0
        buckets[b][2] += true_share
        tot["n"] += 1
        tot["wrong"] += 1 if plur_wrong else 0
    out = {"buckets": {}, "overall": dict(tot)}
    for b in sorted(buckets):
        n, w, ts = buckets[b]
        out["buckets"][str(b)] = {"n": n, "plur_wrong": w,
                                  "plur_wrong_rate": w / n if n else None,
                                  "mean_true_share": ts / n if n else None}
        md(f"- label-share {b:.1f}: n={n} groups, P(plurality wrong)="
           f"{w / n:.3f}, mean true-share {ts / n:.3f}")
    md(f"- overall: {tot['n']} valid groups, P(plurality wrong)="
       f"{tot['wrong'] / tot['n']:.4f}")
    STATS["b2mech"] = out


# ====================================================================== #
# b3stats                                                               #
# ====================================================================== #
def part_b3stats():
    md("\n## 6. B3 标签质量审计（错误分类 / LIMIT 无序 / 宽松匹配上界）\n")
    labels = load_labels()
    meta = load_pool_meta()
    cache = load_cache()
    pool = load_pool(POOL_F)
    out = {}
    # (a) exec_fail taxonomy
    fails = [k for k, v in labels.items() if v == "exec_fail"]
    tax = Counter()
    for k in fails:
        db = meta[k]["db"]
        ns = meta[k]["sql"].strip()
        rec = cache.get(db + "\x00" + ns)
        gold = next((c.gold_sql for c in pool if c.qid == k[0]), "")
        grec = cache.get(db + "\x00" + gold.strip())
        cand_err = (rec and not rec["ok"] and rec["err"]) if rec else "missing"
        gold_err = (grec and not grec["ok"]) if grec else False
        if gold_err:
            tax["gold_fail" + ("+" + str(cand_err) if cand_err else "")] += 1
        elif cand_err:
            tax[str(cand_err)] += 1
        else:
            tax["recovered_ok(now)"] += 1
    out["exec_fail_taxonomy"] = dict(tax)
    out["n_exec_fail"] = len(fails)
    md(f"- exec_fail {len(fails)} 条分类: {dict(tax)}")
    # (b) LIMIT without ORDER BY
    lim_keys = [k for k, v in meta.items()
                if re.search(r"\blimit\b", v["sql"].lower())
                and "order by" not in v["sql"].lower()]
    lim_sqls = sorted({(meta[k]["db"], meta[k]["sql"].strip())
                       for k in lim_keys})
    unstable = 0
    tested = 0
    if lim_sqls:
        def rerun(item):
            db, ns = item
            hs = set()
            for _ in range(5):
                res, err = run_sql_cls(_db_path(db), ns)
                if res is None:
                    return item, None
                hs.add(_result_hashes(res)[0])
            return item, len(hs)

        with ThreadPoolExecutor(max_workers=8) as ex:
            for item, n_h in ex.map(rerun, lim_sqls):
                if n_h is None:
                    continue
                tested += 1
                if n_h > 1:
                    unstable += 1
    n_lim_cand = sum(1 for k in lim_keys)
    out["limit_no_orderby"] = {
        "n_candidates": n_lim_cand,
        "n_questions": len({k[0] for k in lim_keys}),
        "n_unique_sql": len(lim_sqls), "n_tested": tested,
        "n_unstable_5x": unstable}
    md(f"- LIMIT-without-ORDER-BY: {n_lim_cand} candidates "
       f"({len({k[0] for k in lim_keys})} questions, {len(lim_sqls)} unique); "
       f"5×重执行结果不稳定 {unstable}/{tested}")
    # (c) lenient match upper bound on silent labels
    cl, gl = cand_lenient()
    silent = [k for k, v in labels.items() if v == "silent"]
    len_match = sum(1 for k in silent if cl.get(k) and cl[k] == gl.get(k[0]))
    out["lenient_silent"] = {"n_silent": len(silent),
                             "lenient_match": len_match,
                             "upper_bound_rate": len_match / len(silent)}
    md(f"- silent {len(silent)} 条中宽松匹配命中 gold {len_match} 条 "
       f"(假阴性噪声率上界 {len_match / len(silent):.4f})")
    STATS["b3stats"] = out


# ====================================================================== #
# b4stats                                                               #
# ====================================================================== #
def part_b4stats():
    md("\n## 7. B4 统计口径（ECE/Brier 聚类 bootstrap CI + 高区纯度对齐）\n")
    labels = load_labels()
    confs = {"b1": load_conf(B1_F), "b4": load_conf(B4_F),
             "fuse_prod": load_conf(FUSE_F)}
    pa = sorted(load_conf(B4_F))  # Protocol A eval keys (n=4,200)
    out = {}
    for m, conf in confs.items():
        ks = [k for k in pa if k in conf]
        by_q = defaultdict(list)
        for k in ks:
            by_q[k[0]].append(k)
        qs = sorted(by_q)
        co = [labels[k] == "correct" for k in ks]

        def stats_on(keys):
            sc = [conf[k] for k in keys]
            cc = [labels[k] == "correct" for k in keys]
            return (M.ece(sc, cc, 15), M.ece(sc, cc, 10), M.brier(sc, cc))

        e15, e10, br = stats_on(ks)
        rng = random.Random(11)
        boots = {"ece15": [], "ece10": [], "brier": []}
        for _ in range(2000):
            samp = [qs[rng.randrange(len(qs))] for _ in range(len(qs))]
            kk = [k for q in samp for k in by_q[q]]
            a, b, c = stats_on(kk)
            boots["ece15"].append(a)
            boots["ece10"].append(b)
            boots["brier"].append(c)
        ci = {}
        for k, v in boots.items():
            v.sort()
            ci[k] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
        out[m] = {"ece15": e15, "ece10": e10, "brier": br, "ci95": ci,
                  "n": len(ks)}
        md(f"- {m} (n={len(ks)}): ECE15 {e15:.4f} CI95 "
           f"[{ci['ece15'][0]:.4f},{ci['ece15'][1]:.4f}]; ECE10 {e10:.4f} "
           f"[{ci['ece10'][0]:.4f},{ci['ece10'][1]:.4f}]; Brier {br:.4f} "
           f"[{ci['brier'][0]:.4f},{ci['brier'][1]:.4f}]")
    # high-end purity, aligned thresholds, cluster bootstrap CI
    pur = {}
    for m in ("b4", "fuse_prod"):
        conf = confs[m]
        ks = sorted(conf)
        by_q = defaultdict(list)
        for k in ks:
            by_q[k[0]].append(k)
        qs = sorted(by_q)
        for t in (0.85, 0.9):
            sel = [k for k in ks if conf[k] >= t]
            n = len(sel)
            w = sum(1 for k in sel if labels[k] != "correct")
            rng = random.Random(13)
            boots = []
            for _ in range(2000):
                samp = [qs[rng.randrange(len(qs))] for _ in range(len(qs))]
                kk = [k for q in samp for k in by_q[q] if conf[k] >= t]
                if kk:
                    boots.append(sum(1 for k in kk
                                     if labels[k] != "correct") / len(kk))
            boots.sort()
            pur[f"{m}@{t}"] = {
                "n": n, "wrong": w, "rate": w / n if n else None,
                "boot_ci95": [boots[int(0.025 * len(boots))],
                              boots[int(0.975 * len(boots)) - 1]] if boots
                else None}
            r = pur[f"{m}@{t}"]
            md(f"- purity {m}@{t}: n={n}, wrong {w}/{n}"
               + (f" ({w / n * 100:.1f}%), question-bootstrap CI95 "
                  f"[{r['boot_ci95'][0]:.3f},{r['boot_ci95'][1]:.3f}]"
                  if n else " (empty)"))
    out["purity"] = pur
    STATS["b4stats"] = out


# ====================================================================== #
# b1disc                                                                #
# ====================================================================== #
def part_b1disc():
    md("\n## 8. B1 附录披露统计（分布 / unique SQL / Protocol B 构成）\n")
    out = {}
    b1 = load_conf(B1_F)
    b5 = load_conf(os.path.join(os.path.dirname(B1_F), "bird300_ds_b5.jsonl"))
    for m, conf in (("b1", b1), ("b5", b5)):
        dec = Counter(min(9, int(v * 10)) for v in conf.values())
        out[m] = {
            "n": len(conf),
            "deciles": {str(k): dec.get(k, 0) for k in range(10)},
            "ge_095": sum(1 for v in conf.values() if v >= 0.95) / len(conf),
            "eq_1": sum(1 for v in conf.values() if v >= 0.999) / len(conf),
            "mean": sum(conf.values()) / len(conf),
        }
        r = out[m]
        md(f"- {m} (n={r['n']}): mean {r['mean']:.3f}, ≥0.95 "
           f"{r['ge_095'] * 100:.1f}%, =1.0 {r['eq_1'] * 100:.1f}%, "
           f"deciles {r['deciles']}")
    # Protocol B composition: Protocol A eval keys ∩ B2(nucleus10) ∩ B5
    pools = os.path.dirname(B1_F)
    b4 = load_conf(B4_F)
    b2n = load_conf(os.path.join(pools, "bird300_ds_b2_nuc10.jsonl"))
    inter = [k for k in sorted(b4) if k in b2n and k in b5]
    nq = len({k[0] for k in inter})
    src_comp = Counter(k[1] for k in inter)
    out["protocol_b"] = {
        "n_protocol_a": len(b4), "n_b2_nuc10": len(b2n), "n_b5": len(b5),
        "n_intersection": len(inter), "n_questions": nq,
        "sources": dict(src_comp)}
    md(f"- Protocol B = Protocol A eval ∩ B2(nucleus10) ∩ B5 = "
       f"{len(inter)} keys ({nq} questions × {len(inter) // max(1, nq)} "
       f"cands/q, sources {dict(src_comp)}): B2's k=10 consistency vote is "
       f"scored on the nucleus10 decoding group, so the all-method "
       f"intersection is exactly the nucleus10 half of the eval split "
       f"(4,200→2,100); the beam10 half carries B2 shares too (used by the "
       f"fusions) but is not part of the B2 baseline row")
    STATS["b1disc"] = out


# ====================================================================== #
# driftshare                                                            #
# ====================================================================== #
def part_driftshare():
    md("\n## 9. B3 drift 下 share/B2 行为（漂移库上重执行重投票）\n")
    import run_drift as RD
    labels = load_labels()
    shares = {}
    for f in SH_F:
        shares.update(load_conf(f))
    pool = load_pool(POOL_F)
    out = {"variants": []}
    agg_pert = defaultdict(lambda: {"n": 0, "dshare": 0.0, "cands": 0})
    for db, pert in RD.valid_variants():
        dpath = RD.build_one(db, pert)
        precise, col2tables = RD.rewrite_map(db, pert) \
            if pert == "rename" else ({}, {})
        sub = [c for c in pool if c.db_id == db]
        cache_l = {}

        def lab_of(c):
            sql_d = RD.rewrite_sql(c.sql, precise, col2tables) \
                if precise else c.sql
            gold_d = RD.rewrite_sql(c.gold_sql, precise, col2tables) \
                if precise else c.gold_sql
            key = sql_d  # raw key: normalization can merge distinct literals
            if key not in cache_l:
                pred = RD.run_sql(dpath, sql_d)
                gold = RD.run_sql(dpath, gold_d)
                cache_l[key] = ("exec_fail" if pred is None or gold is None
                                else ("correct" if pred == gold else "silent"))
            return cache_l[key]

        labs_d = {c.key: lab_of(c) for c in sub}
        groups = defaultdict(list)
        for c in sub:
            groups[(c.qid, c.source)].append(c.key)
        n = dsum = 0
        sc_orig, sc_drift, co_d = [], [], []
        for g, mks in groups.items():
            valid_d = [labs_d[k] for k in mks if labs_d[k] in ("correct",
                                                               "silent")]
            if not valid_d:
                continue
            sh_d = max(valid_d.count("correct"),
                       valid_d.count("silent")) / len(valid_d)
            sh_o = shares.get(mks[0], 0.0)
            n += 1
            dsum += abs(sh_d - sh_o)
            for k in mks:
                if labs_d[k] != "exec_fail":
                    sc_orig.append(sh_o)
                    sc_drift.append(sh_d)
                    co_d.append(labs_d[k] == "correct")
        row = {"db": db, "pert": pert, "n_groups": n,
               "mean_abs_dshare": dsum / max(1, n),
               "auroc_share_orig_on_drift": M.auroc(sc_orig, co_d),
               "auroc_share_drift_on_drift": M.auroc(sc_drift, co_d)}
        out["variants"].append(row)
        a = agg_pert[pert]
        a["n"] += n
        a["dshare"] += dsum
        a["cands"] += len(co_d)
        md(f"- {db}/{pert}: groups {n}, mean|Δshare| "
           f"{row['mean_abs_dshare']:.3f}, AUROC(share_orig→drift labels) "
           f"{row['auroc_share_orig_on_drift']:.3f}, "
           f"AUROC(share_drift→drift labels) "
           f"{row['auroc_share_drift_on_drift']:.3f}")
        print(f"[driftshare] {db}/{pert} done", flush=True)
    out["by_pert"] = {}
    for pert, a in agg_pert.items():
        out["by_pert"][pert] = {"n_groups": a["n"],
                                "mean_abs_dshare": a["dshare"] / max(1, a["n"])}
    STATS["driftshare"] = out


# ====================================================================== #
# figs                                                                  #
# ====================================================================== #
def part_figs():
    md("\n## 10. 图（C13 pipeline / C14 risk-coverage 带 / B1B5 直方图 / K 敏感）\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, FancyBboxPatch
    OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
          "#56B4E9", "#F0E442", "#000000"]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })
    os.makedirs(FIG_DIR, exist_ok=True)
    out = {}

    # ---- C13: pipeline overview ----
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#000000", fs=7.3, lw=0.9):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06",
            linewidth=lw, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrow(x1, y1, x2, y2, width=0.012,
                                head_width=0.12, head_length=0.12,
                                length_includes_head=True, color="#555555"))

    box(0.2, 5.0, 2.5, 1.1,
        "BIRD dev-300 questions\nDeepSeek-chat\nnucleus10 (T=0.8) + beam10 (T=0.2)\n20 candidates/question (n=6,000)",
        fc="#EAF2FB")
    box(3.6, 5.55, 2.2, 0.55, "execution labeling\ncorrect / silent / exec_fail", fc="#F5F5F5")
    arrow(2.7, 5.55, 3.6, 5.82)
    box(0.2, 3.4, 5.6, 1.1,
        "confidence families (all score the same pool)\n"
        "B0 always-execute · B1 verbalized · B2 consistency share · B3 SAC\n"
        "B4 clause recalibration (13 features + z) · B5 judge · B6/B7 judge variants",
        fc="#FBF0E6")
    arrow(1.5, 5.0, 1.5, 4.5)
    arrow(4.7, 5.55, 4.7, 4.5)
    box(0.2, 2.0, 2.6, 1.0, "Protocol A\nwhite-box intersection\nn=4,200 (eval)", fc="#EAF7EE")
    box(3.2, 2.0, 2.6, 1.0, "Protocol B\nall-method intersection\nn=2,100", fc="#EAF7EE")
    arrow(1.5, 3.4, 1.5, 3.0)
    arrow(4.5, 3.4, 4.5, 3.0)
    box(6.4, 3.9, 3.4, 1.0,
        "fusion of B4 × share\nproduct / min / L2 stack\n(zero extra LLM calls)", fc="#FBEAF2")
    arrow(5.8, 4.3, 6.4, 4.3)
    box(6.4, 2.4, 3.4, 1.0,
        "B8 split-conformal wrapper\nα=0.02, 20 seeds\nquestion-level split", fc="#FBEAF2")
    arrow(8.1, 3.9, 8.1, 3.4)
    box(0.2, 0.3, 9.6, 1.2,
        "evaluation axes:  AUROC / ECE / risk-coverage / selective execution  ·  "
        "per-question deployment metrics\n"
        "5 refit seeds + cluster bootstrap  ·  LODO transfer  ·  schema drift "
        "(rename / add_col / split_tbl)  ·  conformal leakage audit",
        fc="#F2F2F2")
    arrow(1.5, 2.0, 1.5, 1.5)
    arrow(4.5, 2.0, 4.5, 1.5)
    arrow(8.1, 2.4, 8.1, 1.5)
    fig.savefig(os.path.join(FIG_DIR, "fig_pipeline.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_pipeline.png"), dpi=300)
    plt.close(fig)
    out["pipeline"] = "ok"

    # ---- C14: risk-coverage 5-seed band ----
    from p0_reanalysis import _fit_b4_seed
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
        z = math.log(max(math.exp(sum(c.token_logprobs)
                                  if c.token_logprobs else 0.0), 1e-12))
        data.append((c.key, z, clause_features(c.sql), lab == "correct",
                     c.qid))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    curves = {"b4": [], "fuse_prod": []}
    for seed in range(5):
        sc = _fit_b4_seed(data, seed)
        ks = sorted(sc)
        co = [labels[k] == "correct" for k in ks]
        fsc = {k: sc[k] * shares.get(k, 0.5) for k in ks}
        for m, cd in (("b4", sc), ("fuse_prod", fsc)):
            cov, risk = M.risk_coverage([cd[k] for k in ks], co)
            # downsample to ~120 points
            step = max(1, len(cov) // 120)
            curves[m].append((cov[::step], risk[::step]))
    for ax, m, c in zip(axes, ("b4", "fuse_prod"), (OI[0], OI[1])):
        n_pts = min(len(c[0]) for c in curves[m])
        risks = []
        for cov, risk in curves[m]:
            risks.append(np_interpolate(cov, risk, n_pts))
        import numpy as np
        R = np.array(risks)
        cov0 = curves[m][0][0]
        xs = np.linspace(cov0[0], cov0[-1], n_pts)
        mean = R.mean(axis=0)
        lo, hi = R.min(axis=0), R.max(axis=0)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.22, label="min–max (5 seeds)")
        ax.plot(xs, mean, color=c, lw=1.4, label="mean")
        ax.set_xlabel("coverage")
        ax.set_title("B4" if m == "b4" else "fuse$_{prod}$")
        ax.set_xlim(0.02, 1.0)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("selective risk")
    fig.savefig(os.path.join(FIG_DIR, "fig_rc_band.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_rc_band.png"), dpi=300)
    plt.close(fig)
    out["rc_band"] = "ok"

    # ---- B1/B5 histograms ----
    b1 = load_conf(B1_F)
    b5 = load_conf(os.path.join(os.path.dirname(B1_F), "bird300_ds_b5.jsonl"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, (m, conf) in zip(axes, (("B1 verbalized", b1),
                                    ("B5 judge", b5))):
        vals = [v for v in conf.values()]
        ax.hist(vals, bins=20, range=(0, 1), color=OI[3],
                edgecolor="white", linewidth=0.4)
        sat = sum(1 for v in vals if v >= 0.999) / len(vals)
        ax.set_title(f"{m} (n={len(vals):,})")
        ax.set_xlabel("confidence")
        ax.text(0.97, 0.72, f"{sat * 100:.0f}% at 1.0", transform=ax.transAxes,
                ha="right", fontsize=8)
    axes[0].set_ylabel("candidates")
    fig.savefig(os.path.join(FIG_DIR, "fig_b1b5_hist.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig_b1b5_hist.png"), dpi=300)
    plt.close(fig)
    out["b1b5_hist"] = "ok"

    # ---- K-sensitivity curve ----
    if "ksens" in STATS or os.path.exists(OUT_JSON):
        ks_stats = STATS.get("ksens")
        if ks_stats is None:
            try:
                with open(OUT_JSON, encoding="utf-8") as f:
                    ks_stats = json.load(f).get("ksens")
            except (json.JSONDecodeError, OSError):
                ks_stats = None
        if ks_stats:
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            xs = [5, 10, 20]
            fuse = [ks_stats["K5"]["fuse_auroc"]["mean"],
                    ks_stats["K10"]["fuse_auroc"],
                    ks_stats["K20_pooled"]["fuse_auroc"]]
            fuse_sd = [ks_stats["K5"]["fuse_auroc"]["sd"], 0.0, 0.0]
            b2 = [ks_stats["K5"]["b2_auroc"]["mean"],
                  ks_stats["K10"]["b2_auroc"],
                  ks_stats["K20_pooled"]["b2_auroc"]]
            b2_sd = [ks_stats["K5"]["b2_auroc"]["sd"], 0.0, 0.0]
            ax.errorbar(xs, fuse, yerr=fuse_sd, color=OI[0], marker="o",
                        ms=4, capsize=3, lw=1.3, label="fuse$_{prod}$ AUROC")
            ax.errorbar(xs, b2, yerr=b2_sd, color=OI[1], marker="s",
                        ms=4, capsize=3, lw=1.3, label="share (B2) AUROC")
            ax.set_xticks(xs)
            ax.set_xticklabels(["5", "10 (default)", "20 (pooled)"])
            ax.set_xlabel("voting pool size $K$")
            ax.set_ylabel("AUROC")
            ax.legend()
            fig.savefig(os.path.join(FIG_DIR, "fig_ksens.pdf"))
            fig.savefig(os.path.join(FIG_DIR, "fig_ksens.png"), dpi=300)
            plt.close(fig)
            out["ksens"] = "ok"
    for k, v in out.items():
        md(f"- {k}: {v}")
    STATS["figs"] = out


def np_interpolate(cov, risk, n_pts):
    """Resample a risk-coverage curve to n common coverage points."""
    import numpy as np
    xs = np.linspace(cov[0], cov[-1], n_pts)
    return np.interp(xs, cov, risk)


# ====================================================================== #
# main                                                                  #
# ====================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    args = ap.parse_args()
    p = args.part
    parts = {
        "execache": part_execache, "cfix": part_cfix, "lodoref": part_lodoref,
        "ksens": part_ksens, "select": part_select, "b2mech": part_b2mech,
        "b3stats": part_b3stats, "b4stats": part_b4stats,
        "b1disc": part_b1disc, "driftshare": part_driftshare,
        "figs": part_figs,
    }
    order = ["execache", "cfix", "lodoref", "ksens", "select", "b2mech",
             "b3stats", "b4stats", "b1disc", "driftshare", "figs"]
    if p == "all":
        run = order
    else:
        run = [x.strip() for x in p.split(",") if x.strip()]
    for name in run:
        print(f"=== part {name} ===", flush=True)
        parts[name]()
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
