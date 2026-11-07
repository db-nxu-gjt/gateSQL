# -*- coding: utf-8 -*-
"""gen_excel.py — 将 GATE-SQL 全部实验结果重算并写入 Excel（paper/results/）。

产出（每实验一个 xlsx，多 sheet）:
  main_table.xlsx      口径 A/B 主表（Phase 2/3/5）
  risk_coverage.xlsx   risk-coverage 曲线数据
  difficulty.xlsx      按难度分解
  C1_lodo.xlsx         leave-one-db-out 11 折明细 + macro
  C2_drift.xlsx        schema 漂移层 1（24 变体）+ 层 2（判官）
  conformal_ks.xlsx    20 seeds + KS 交换性检验
  laya_triangle.xlsx   Laya 三角负结果
  phase1_archive.xlsx  AUC 归档池（动机/附录用）

复用 baseline 现有实现重算（避免手抄错误），结果与 results/*.md 一致。
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "tools"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
from common import metrics as M  # noqa: E402
from common.schema import load_labels, load_pool  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402
import run_drift as RD  # noqa: E402

OUT = os.path.normpath(os.path.join(_ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
POOLDIR = os.path.join(_ROOT, "data", "pools")

HDR_FILL = PatternFill("solid", fgColor="1F4E79")
HDR_FONT = Font(bold=True, color="FFFFFF", size=11)
ALT_FILL = PatternFill("solid", fgColor="DDEBF7")
THIN = Side(style="thin", color="B0B0B0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def write_sheet(wb, title, header, rows, widths=None, note=None):
    ws = wb.create_sheet(title)
    r0 = 1
    if note:
        ws.cell(1, 1, note).font = Font(italic=True, size=9, color="606060")
        r0 = 3
    for j, h in enumerate(header, 1):
        c = ws.cell(r0, j, h)
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, BORDER
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
    for i, row in enumerate(rows):
        for j, v in enumerate(row, 1):
            c = ws.cell(r0 + 1 + i, j, v)
            c.border = BORDER
            if isinstance(v, float):
                c.number_format = "0.000"
            if i % 2 == 1:
                c.fill = ALT_FILL
    for j, h in enumerate(header, 1):
        w = (widths[j - 1] if widths and j - 1 < len(widths)
             else max(10, min(34, len(str(h)) + 6)))
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(r0 + 1, 1)
    return ws


def load_conf(fn):
    out = {}
    with open(os.path.join(POOLDIR, fn), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    float(r["conf"])
    return out


def load_labels_raw():
    labels = {}
    with open(os.path.join(POOLDIR, "bird300_ds_labels.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    r["label"]
    return labels


METHODS = {
    "B1 Verbalized": "bird300_ds_b1.jsonl",
    "B2 Consistency (k=10)": "bird300_ds_b2_nuc10.jsonl",
    "B3 SAC-Avg": "bird300_ds_b3_sac.jsonl",
    "B4 Clause Recalib": "bird300_ds_b4.jsonl",
    "B5 LLM Judge": "bird300_ds_b5.jsonl",
    "Fuse-min": "bird300_ds_fuse_min.jsonl",
    "Fuse-prod": "bird300_ds_fuse_prod.jsonl",
    "Fuse-stack (L2)": "bird300_ds_fuse_stack.jsonl",
}
WHITE = ["B1 Verbalized", "B3 SAC-Avg", "B4 Clause Recalib",
         "Fuse-min", "Fuse-prod", "Fuse-stack (L2)"]


def sel_ex_at(cf, keys, labels, cov):
    # 确定性平局处理：分数并列时按 key 升序（B1 等粗粒度分数下 set 顺序不稳定）
    ranked = sorted(keys, key=lambda k: (-cf[k], k))
    take = ranked[: max(1, int(len(ranked) * cov))]
    return sum(labels[k] == "correct" for k in take) / len(take)


def risk_coverage(labels, confs, keys):
    rows = []
    for cov in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        r = [cov]
        for n in ("B1 Verbalized", "B4 Clause Recalib", "Fuse-min",
                  "Fuse-prod", "Fuse-stack (L2)"):
            r.append(sel_ex_at(confs[n], keys, labels, cov))
        rows.append(r)
    return rows


def main() -> None:
    labels = load_labels_raw()
    confs = {n: load_conf(fn) for n, fn in METHODS.items()}

    # ---------- 口径 A / B ----------
    keysA = None
    for n in ("B1 Verbalized", "B3 SAC-Avg", "B4 Clause Recalib",
              "Fuse-prod"):
        ks = set(confs[n]) & set(labels)
        keysA = ks if keysA is None else keysA & ks
    keysB = set(keysA)
    for n in ("B2 Consistency (k=10)", "B5 LLM Judge"):
        keysB &= set(confs[n]) & set(labels)

    wb = Workbook()
    wb.remove(wb.active)
    wb_c1 = Workbook()
    wb_c1.remove(wb_c1.active)
    wb_c2 = Workbook()
    wb_c2.remove(wb_c2.active)
    wb_cf = Workbook()
    wb_cf.remove(wb_cf.active)
    wb_ly = Workbook()
    wb_ly.remove(wb_ly.active)
    wb_p1 = Workbook()
    wb_p1.remove(wb_p1.active)

    for tag, keys in (("A (white-box, n=%d)" % len(keysA), keysA),
                      ("B (all methods, n=%d)" % len(keysB), keysB)):
        rows = []
        names = WHITE if tag.startswith("A") else list(METHODS)
        for n in names:
            ks = keys if tag.startswith("B") else keysA
            co = [labels[k] == "correct" for k in ks]
            sc = [confs[n][k] for k in ks]
            rows.append([n, len(ks), M.auroc(sc, co), M.ece(sc, co),
                        sel_ex_at(confs[n], ks, labels, 0.5),
                        sel_ex_at(confs[n], ks, labels, 0.3)])
        ex = sum(labels[k] == "correct" for k in keysA) / len(keysA)
        rows.append(["B0 Always-Execute", len(keysA), None, None, ex, ex])
        write_sheet(wb, "Main-%s" % tag[0],
                    ["Method", "n", "AUROC", "ECE15", "sel_EX@50%",
                     "sel_EX@30%"],
                    rows, note="Coverage-selective execution on identical "
                               "candidate sets; exec_fail counted as wrong.")

    # ---------- risk-coverage ----------
    write_sheet(wb, "RiskCov", ["Coverage", "B1", "B4", "Fuse-min",
                                "Fuse-prod", "Fuse-stack"],
                risk_coverage(labels, confs, keysA),
                note="sel_EX at coverage on set A (n=%d, incl. exec_fail); "
                     "B0 EX=0.537." % len(keysA))

    # ---------- difficulty ----------
    dev = json.load(open(os.path.join(_ROOT, "data", "raw", "bird",
                                      "dev.json"), encoding="utf-8"))
    diff_of = {i: it.get("difficulty", "") for i, it in enumerate(dev)}
    rows = []
    for d in ("simple", "moderate", "challenging"):
        keys = [k for k in keysA if diff_of.get(k[0]) == d]
        r = [d, len(keys), sum(labels[k] == "correct" for k in keys) / len(keys)]
        for n in ("B1 Verbalized", "B4 Clause Recalib", "Fuse-prod"):
            co = [labels[k] == "correct" for k in keys]
            r.append(M.auroc([confs[n][k] for k in keys], co))
            r.append(sel_ex_at(confs[n], keys, labels, 0.5))
        rows.append(r)
    write_sheet(wb, "Difficulty",
                ["Difficulty", "n", "EX", "B1 AUROC", "B1 sel_EX@50",
                 "B4 AUROC", "B4 sel_EX@50", "Fuse-prod AUROC",
                 "Fuse-prod sel_EX@50"],
                rows, note="AUROC / selective EX@50% per BIRD difficulty.")

    # ---------- C1 LODO（重算） ----------
    wb1 = wb_c1
    pool = load_pool(os.path.join(POOLDIR, "bird300_ds_pool.jsonl"),
                     require_logprobs=True)
    lab = load_labels(os.path.join(POOLDIR, "bird300_ds_labels.jsonl"))
    data = []
    for c in pool:
        l = lab.get(c.key)
        if l is None:
            continue
        z = sum(c.token_logprobs) if c.token_logprobs else 0.0
        data.append((c, z, clause_features(c.sql), l["correct"]))
    shares = {}
    for s in ("nuc10", "beam10"):
        shares.update(load_conf(f"bird300_ds_b2_{s}.jsonl"))
    b1c = load_conf("bird300_ds_b1.jsonl")
    b3c = load_conf("bird300_ds_b3_sac.jsonl")

    def logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    dbs = sorted({d[0].db_id for d in data})
    lodo_rows = []
    agg = {}
    for hold in dbs:
        te = [i for i, d in enumerate(data) if d[0].db_id == hold]
        tr = [i for i, d in enumerate(data) if d[0].db_id != hold]
        dim = len(data[0][2])
        mu = [sum(data[i][2][j] for i in tr) / len(tr) for j in range(dim)]
        sd = [max(1e-6, math.sqrt(sum((data[i][2][j] - mu[j]) ** 2
                                      for i in tr) / len(tr)))
              for j in range(dim)]

        def mk(i):
            return ([(f - m) / s for f, m, s in zip(data[i][2], mu, sd)]
                    + [data[i][1] / 10.0])

        p1 = fit_logistic([mk(i) for i in tr], [data[i][3] for i in tr])

        def raw(i):
            x = mk(i)
            zz = p1[-1] + sum(w * xj for w, xj in zip(p1[:-1], x))
            return 1 / (1 + math.exp(-max(-30, min(30, zz))))

        tq = sorted({data[i][0].qid for i in tr})
        half = max(1, len(tq) // 2)
        oof = {}
        for a, b in ((set(tq[:half]), set(tq[half:])),
                     (set(tq[half:]), set(tq[:half]))):
            tri = [i for i in tr if data[i][0].qid in a]
            tei = [i for i in tr if data[i][0].qid in b]
            pp = fit_logistic([mk(i) for i in tri],
                              [data[i][3] for i in tri])
            for i in tei:
                x = mk(i)
                zz = pp[-1] + sum(w * xj for w, xj in zip(pp[:-1], x))
                oof[i] = 1 / (1 + math.exp(-max(-30, min(30, zz))))
        l2f = [i for i in tr if i in oof]
        l2r = [[logit(oof[i]), shares.get(data[i][0].key, 0.5),
                logit(oof[i]) * shares.get(data[i][0].key, 0.5)]
               for i in l2f]
        l2y = [1.0 if data[i][3] else 0.0 for i in l2f]
        d2 = 3
        mu2 = [sum(r[j] for r in l2r) / len(l2r) for j in range(d2)]
        sd2 = [max(1e-6, math.sqrt(sum((r[j] - mu2[j]) ** 2 for r in l2r)
                                  / len(l2r))) for j in range(d2)]
        p2 = fit_logistic([[(r[j] - mu2[j]) / sd2[j] for j in range(d2)]
                           for r in l2r], l2y)

        def fuse(i):
            s = raw(i)
            sh = shares.get(data[i][0].key, 0.5)
            x = [logit(s), sh, logit(s) * sh]
            zz = p2[-1] + sum(w * ((xj - m) / s_)
                              for w, xj, m, s_ in zip(p2[:-1], x, mu2, sd2))
            return 1 / (1 + math.exp(-max(-30, min(30, zz))))

        co = [data[i][3] for i in te]
        k2i = {data[i][0].key: i for i in range(len(data))}
        vals = {
            "B1": M.auroc([b1c.get(data[i][0].key, 0.5) for i in te], co),
            "B3": M.auroc([b3c[data[i][0].key] for i in te
                           if data[i][0].key in b3c],
                          [data[k2i[data[i][0].key]][3] for i in te
                           if data[i][0].key in b3c]),
            "B4": M.auroc([raw(i) for i in te], co),
            "Fuse-prod": M.auroc([fuse(i) for i in te], co),
            "B4 ECE": M.ece([raw(i) for i in te], co),
            "Fuse ECE": M.ece([fuse(i) for i in te], co),
        }
        for k, v in vals.items():
            agg.setdefault(k, []).append(v)
        lodo_rows.append([hold, len(te)] +
                         [round(vals[k], 4) for k in
                          ("B1", "B3", "B4", "Fuse-prod",
                           "B4 ECE", "Fuse ECE")])
    import statistics as st
    macro = ["MACRO (11 folds)", None]
    macro += [round(st.mean(agg[k]), 4) for k in
              ("B1", "B3", "B4", "Fuse-prod", "B4 ECE", "Fuse ECE")]
    lodo_rows.append(macro)
    json.dump(lodo_rows, open(os.path.join(OUT, "_lodo_rows.json"), "w"),
              ensure_ascii=False)
    write_sheet(wb1, "C1-LODO",
                ["Held-out DB", "n", "B1", "B3", "B4-LODO", "Fuse-LODO",
                 "B4 ECE", "Fuse ECE"],
                lodo_rows,
                note="Leave-one-db-out: refit B4 logistic & fuse L2 on "
                     "other DBs per fold.")

    # ---------- C2 层1（重算执行标注） ----------
    dmodel = RD._global_b4_model()
    data_g, raw_of, fuse_of = dmodel
    drift_rows = []
    for db, pert in RD.valid_variants():
        dpath = RD.build_one(db, pert)
        precise, c2t = RD.rewrite_map(db, pert) if pert == "rename" \
            else ({}, {})
        sub = [d for d in data_g if d[0].db_id == db]
        agree = n = 0
        sc4, scf, co = [], [], []
        for c, z, feat, corr in sub:
            sql_d = RD.rewrite_sql(c.sql, precise, c2t) if precise else c.sql
            gold_d = (RD.rewrite_sql(c.gold_sql, precise, c2t)
                      if precise else c.gold_sql)
            pred = RD.run_sql(dpath, sql_d)
            if pred is None:
                lab_d = "exec_fail"
            else:
                g = RD.run_sql(dpath, gold_d)
                lab_d = "exec_fail" if g is None else (
                    "correct" if pred == g else "silent")
            n += 1
            agree += lab_d == labels.get(c.key)
            f_d = clause_features(sql_d)
            sc4.append(raw_of(f_d, z))
            scf.append(fuse_of(f_d, z, shares.get(c.key, 0.5)))
            co.append(lab_d == "correct")
        co_orig = [corr for _, _, _, corr in sub]
        drift_rows.append([f"{db}/{pert}", n, agree / n,
                           M.auroc(sc4, co),
                           M.auroc([raw_of(f, z) for _, z, f, _ in sub],
                                   co_orig),
                           M.auroc(scf, co),
                           M.auroc([fuse_of(f, z, shares.get(c2.key, 0.5))
                                    for c2, z, f, _ in sub], co_orig),
                           M.ece(sc4, co)])
        print(f"[c2-layer1] {db}/{pert} done", flush=True)
    json.dump(drift_rows, open(os.path.join(OUT, "_drift_rows.json"), "w"),
              ensure_ascii=False)

    # ---------- C2 层2（判官，从 jsonl） ----------
    drows = [json.loads(l) for l in open(
        os.path.join(POOLDIR, "bird300_ds_b1_drift.jsonl"),
        encoding="utf-8") if l.strip()]
    rawlab = load_labels(os.path.join(POOLDIR, "bird300_ds_labels.jsonl"))
    for r in drows:
        l = rawlab.get((int(r["qid"]), str(r["source"]), int(r["cand_idx"])))
        r["label_orig"] = l["raw"] if l else None
    b1_rows = []
    for pert in ("rename", "split_tbl"):
        sub = [r for r in drows if r["pert"] == pert
               and r.get("conf_orig") is not None
               and r.get("label_orig") in ("correct", "silent")]
        co = [r["label_orig"] == "correct" for r in sub]
        b1_rows.append([pert, len(sub), M.auroc([r["conf_orig"] for r in sub],
                                                co),
                        M.auroc([r["conf"] for r in sub], co),
                        M.ece([r["conf_orig"] for r in sub], co),
                        M.ece([r["conf"] for r in sub], co),
                        sum(abs(r["conf"] - r["conf_orig"])
                            for r in sub) / len(sub)])

    write_sheet(wb_c2, "C2-Exec",
                ["Variant", "n", "Label agreement",
                 "B4-drift AUROC", "B4 orig AUROC",
                 "Fuse-drift AUROC", "Fuse orig AUROC", "B4-drift ECE"],
                drift_rows,
                note="Layer 1: candidates re-labelled on drifted sqlite; "
                     "B4/fuse re-scored with original global model.")
    write_sheet(wb_c2, "C2-Judge",
                ["Perturbation", "n", "AUROC orig", "AUROC drift",
                 "ECE orig", "ECE drift", "mean |dconf|"],
                b1_rows,
                note="Layer 2: B1 verbalized judge re-queried with drifted "
                     "DDL prompt (DeepSeek).")

    # ---------- conformal 20 seeds + KS（重算） ----------
    from scipy import stats as sps
    conf_rows = []
    for name, fn in (("B4", "bird300_ds_b4.jsonl"),
                     ("Fuse-prod", "bird300_ds_fuse_prod.jsonl")):
        cf = load_conf(fn)
        keys = sorted(set(cf) & set(labels))
        qids = sorted({k[0] for k in keys})
        n_hold = 0
        leaks, ks1p, ks2p, covs = [], [], [], []
        per = []
        for seed in range(20):
            rq = qids[:]
            random.Random(seed).shuffle(rq)
            cal_q = set(rq[: max(1, int(len(rq) * 0.4))])
            calw = sorted(1 - cf[k] for k in keys
                          if k[0] in cal_q and labels[k] != "correct")
            test = [(k, cf[k], labels[k] != "correct") for k in keys
                    if k[0] not in cal_q]
            n_c = len(calw)
            acc = [(s, bad) for k, s, bad in test
                   if (1 + bisect.bisect_right(calw, 1 - s)) / (n_c + 1)
                   <= 0.02]
            tw = [1 - s for _, s, bad in test if bad]
            aw = sum(1 for _, bad in acc if bad)
            leak = aw / max(1, len(tw))
            cov = len(acc) / len(test)
            holds = leak <= 0.02 + 1e-9
            n_hold += int(holds)
            ks1 = sps.ks_2samp(calw, tw)
            pv = [(1 + bisect.bisect_right(calw, s)) / (n_c + 1)
                  for s in tw]
            ks2 = sps.kstest(pv, "uniform")
            leaks.append(leak)
            ks1p.append(ks1.pvalue)
            ks2p.append(ks2.pvalue)
            covs.append(cov)
            per.append([seed, round(leak, 4), holds, round(cov, 4),
                        round(ks1.pvalue, 5), round(ks2.pvalue, 5)])
        conf_rows.append([name, 20, int(n_hold),
                          round(sum(leaks) / 20, 4),
                          round(sorted(leaks)[10], 4),
                          round(max(leaks) / 0.02, 1),
                          int(sum(p < 0.05 for p in ks1p)),
                          int(sum(p < 0.05 for p in ks2p)), per])
    json.dump(conf_rows, open(os.path.join(OUT, "_conf_rows.json"), "w"),
              ensure_ascii=False)
    ws = write_sheet(wb_cf, "Conformal-KS",
                     ["Score", "seeds", "holds", "leak mean", "leak median",
                      "max leak/alpha", "KS1 rejects", "KS2 rejects"],
                     [r[:8] for r in conf_rows],
                     note="alpha=0.02, cal=40% questions, seeds 0-19. "
                          "KS1: cal_wrong vs test_wrong score dist; "
                          "KS2: test_wrong conformal p-values vs U(0,1).")
    for i, r in enumerate(conf_rows):
        write_sheet(wb_cf, f"Conf-{r[0]}-seeds",
                    ["seed", "leak", "holds", "coverage",
                     "KS1 p", "KS2 p"], r[8])

    # ---------- Laya 三角 + Phase 1（核对过的存档数字） ----------
    write_sheet(wb_ly, "Laya-Triangle",
                ["Setting", "Executor", "Pool", "AUROC", "ECE15"],
                [["B6 zero-shot", "Local Laya base", "AUC archive",
                  0.5605, 0.3722],
                 ["B7 FT+temperature", "Local Laya base", "AUC archive",
                  0.5032, 0.6053],
                 ["B6-DS zero-shot", "DeepSeek-chat API", "bird300_ds",
                  0.537, 0.338]],
                note="7-question Laya protocol is weak across executors "
                     "and pools; claimed iteration (ECE 0.081) unarchived.")
    write_sheet(wb_p1, "Phase1-Archive",
                ["Method", "n", "AUROC", "ECE15", "Note"],
                [["B1 Verbalized (DS)", 2174, 0.614, 0.328,
                  "strongest on heterogeneous pool"],
                 ["B6 Laya zero-shot", 2041, 0.561, 0.372, ""],
                 ["B7 Laya FT+temp", 2041, 0.503, 0.605,
                  "calibration worsened"],
                 ["B8 conformal on B7", "-", "-", "-",
                  "guarantee holds; coverage 1.55%"]],
                note="AUC archived heterogeneous pool (motivation/appendix "
                     "evidence only; protocols differ from main pool).")

    wb.save(os.path.join(OUT, "main_table.xlsx"))
    wb_c1.save(os.path.join(OUT, "C1_lodo.xlsx"))
    wb_c2.save(os.path.join(OUT, "C2_drift.xlsx"))
    wb_cf.save(os.path.join(OUT, "conformal_ks.xlsx"))
    wb_ly.save(os.path.join(OUT, "laya_triangle.xlsx"))
    wb_p1.save(os.path.join(OUT, "phase1_archive.xlsx"))
    for f in ("main_table", "C1_lodo", "C2_drift", "conformal_ks",
              "laya_triangle", "phase1_archive"):
        print("saved:", os.path.join(OUT, f + ".xlsx"))


if __name__ == "__main__":
    main()
