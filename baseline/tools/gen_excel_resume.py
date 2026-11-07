# -*- coding: utf-8 -*-
"""gen_excel_resume.py — gen_excel.py 断点续跑。

_lodo_rows.json / _drift_rows.json 已由上次运行落盘（C1 与 C2 层1 计算完成），
本脚本跳过这两段重计算，重算其余便宜部分并保存全部 6 个 xlsx。
"""
from __future__ import annotations

import bisect
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "tools"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
import gen_excel as GE  # noqa: E402
from common import metrics as M  # noqa: E402
from common.schema import load_labels  # noqa: E402

OUT = GE.OUT
POOLDIR = GE.POOLDIR


def main() -> None:
    labels = GE.load_labels_raw()
    confs = {n: GE.load_conf(fn) for n, fn in GE.METHODS.items()}
    lodo_rows = json.load(open(os.path.join(OUT, "_lodo_rows.json"),
                               encoding="utf-8"))
    drift_rows = json.load(open(os.path.join(OUT, "_drift_rows.json"),
                                encoding="utf-8"))
    print("[resume] loaded _lodo_rows/_drift_rows", flush=True)

    # ---------- 口径 A / B ----------
    keysA = None
    for n in ("B1 Verbalized", "B3 SAC-Avg", "B4 Clause Recalib",
              "Fuse-prod"):
        ks = set(confs[n]) & set(labels)
        keysA = ks if keysA is None else keysA & ks
    keysB = set(keysA)
    for n in ("B2 Consistency (k=10)", "B5 LLM Judge"):
        keysB &= set(confs[n]) & set(labels)

    wb = GE.Workbook(); wb.remove(wb.active)
    wb_c1 = GE.Workbook(); wb_c1.remove(wb_c1.active)
    wb_c2 = GE.Workbook(); wb_c2.remove(wb_c2.active)
    wb_cf = GE.Workbook(); wb_cf.remove(wb_cf.active)
    wb_ly = GE.Workbook(); wb_ly.remove(wb_ly.active)
    wb_p1 = GE.Workbook(); wb_p1.remove(wb_p1.active)

    for tag, keys in (("A (white-box, n=%d)" % len(keysA), keysA),
                      ("B (all methods, n=%d)" % len(keysB), keysB)):
        rows = []
        names = GE.WHITE if tag.startswith("A") else list(GE.METHODS)
        for n in names:
            ks = keys if tag.startswith("B") else keysA
            co = [labels[k] == "correct" for k in ks]
            sc = [confs[n][k] for k in ks]
            rows.append([n, len(ks), M.auroc(sc, co), M.ece(sc, co),
                        GE.sel_ex_at(confs[n], ks, labels, 0.5),
                        GE.sel_ex_at(confs[n], ks, labels, 0.3)])
        ex = sum(labels[k] == "correct" for k in keysA) / len(keysA)
        rows.append(["B0 Always-Execute", len(keysA), None, None, ex, ex])
        GE.write_sheet(wb, "Main-%s" % tag[0],
                       ["Method", "n", "AUROC", "ECE15", "sel_EX@50%",
                        "sel_EX@30%"],
                       rows, note="Coverage-selective execution on identical "
                                  "candidate sets; exec_fail counted as wrong.")
        print("[main]", tag, flush=True)

    GE.write_sheet(wb, "RiskCov", ["Coverage", "B1", "B4", "Fuse-min",
                                   "Fuse-prod", "Fuse-stack"],
                   GE.risk_coverage(labels, confs, keysA),
                   note="sel_EX at coverage on set A (n=%d, incl. exec_fail); "
                        "B0 EX=0.537." % len(keysA))

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
            r.append(GE.sel_ex_at(confs[n], keys, labels, 0.5))
        rows.append(r)
    GE.write_sheet(wb, "Difficulty",
                   ["Difficulty", "n", "EX", "B1 AUROC", "B1 sel_EX@50",
                    "B4 AUROC", "B4 sel_EX@50", "Fuse-prod AUROC",
                    "Fuse-prod sel_EX@50"],
                   rows, note="AUROC / selective EX@50% per BIRD difficulty.")

    # ---------- C1（读中转 json） ----------
    GE.write_sheet(wb_c1, "C1-LODO",
                   ["Held-out DB", "n", "B1", "B3", "B4-LODO", "Fuse-LODO",
                    "B4 ECE", "Fuse ECE"],
                   lodo_rows,
                   note="Leave-one-db-out: refit B4 logistic & fuse L2 on "
                        "other DBs per fold.")

    # ---------- C2 层1（读中转 json） ----------
    GE.write_sheet(wb_c2, "C2-Exec",
                   ["Variant", "n", "Label agreement",
                    "B4-drift AUROC", "B4 orig AUROC",
                    "Fuse-drift AUROC", "Fuse orig AUROC", "B4-drift ECE"],
                   drift_rows,
                   note="Layer 1: candidates re-labelled on drifted sqlite; "
                        "B4/fuse re-scored with original global model.")

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
    GE.write_sheet(wb_c2, "C2-Judge",
                   ["Perturbation", "n", "AUROC orig", "AUROC drift",
                    "ECE orig", "ECE drift", "mean |dconf|"],
                   b1_rows,
                   note="Layer 2: B1 verbalized judge re-queried with drifted "
                        "DDL prompt (DeepSeek).")

    # ---------- conformal 20 seeds + KS（重算，快速） ----------
    from scipy import stats as sps
    conf_rows = []
    for name, fn in (("B4", "bird300_ds_b4.jsonl"),
                     ("Fuse-prod", "bird300_ds_fuse_prod.jsonl")):
        cf = GE.load_conf(fn)
        keys = sorted(set(cf) & set(labels))
        qids = sorted({k[0] for k in keys})
        n_hold = 0
        leaks, ks1p, ks2p, covs = [], [], [], []
        per = []
        for seed in range(20):
            rq = qids[:]
            import random
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
        print("[conformal]", name, "done", flush=True)
    json.dump(conf_rows, open(os.path.join(OUT, "_conf_rows.json"), "w"),
              ensure_ascii=False)
    GE.write_sheet(wb_cf, "Conformal-KS",
                   ["Score", "seeds", "holds", "leak mean", "leak median",
                    "max leak/alpha", "KS1 rejects", "KS2 rejects"],
                   [r[:8] for r in conf_rows],
                   note="alpha=0.02, cal=40% questions, seeds 0-19. "
                        "KS1: cal_wrong vs test_wrong score dist; "
                        "KS2: test_wrong conformal p-values vs U(0,1).")
    for r in conf_rows:
        GE.write_sheet(wb_cf, "Conf-%s-seeds" % r[0],
                       ["seed", "leak", "holds", "coverage",
                        "KS1 p", "KS2 p"], r[8])

    # ---------- Laya 三角 + Phase 1（核对过的存档数字） ----------
    GE.write_sheet(wb_ly, "Laya-Triangle",
                   ["Setting", "Executor", "Pool", "AUROC", "ECE15"],
                   [["B6 zero-shot", "Local Laya base", "AUC archive",
                     0.5605, 0.3722],
                    ["B7 FT+temperature", "Local Laya base", "AUC archive",
                     0.5032, 0.6053],
                    ["B6-DS zero-shot", "DeepSeek-chat API", "bird300_ds",
                     0.537, 0.338]],
                   note="7-question Laya protocol is weak across executors "
                        "and pools; claimed iteration (ECE 0.081) unarchived.")
    GE.write_sheet(wb_p1, "Phase1-Archive",
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
        print("saved:", os.path.join(OUT, f + ".xlsx"), flush=True)


if __name__ == "__main__":
    main()
