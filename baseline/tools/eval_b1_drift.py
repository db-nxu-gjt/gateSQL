# -*- coding: utf-8 -*-
"""B1 判官漂移结果评测（层 2 汇总）。"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
from common import metrics as M  # noqa: E402
from common.schema import load_labels  # noqa: E402

rows = [json.loads(l) for l in open(
    os.path.join(_ROOT, "data", "pools", "bird300_ds_b1_drift.jsonl"),
    encoding="utf-8") if l.strip()]
labels = load_labels(os.path.join(_ROOT, "data", "pools",
                                  "bird300_ds_labels.jsonl"))
for r in rows:
    lab = labels.get((int(r["qid"]), str(r["source"]), int(r["cand_idx"])))
    r["label_orig"] = lab["raw"] if lab else None
print(f"total {len(rows)}")
for pert in ("rename", "split_tbl"):
    sub = [r for r in rows if r["pert"] == pert
           and r.get("conf_orig") is not None
           and r.get("label_orig") in ("correct", "silent")]
    if not sub:
        print(f"{pert}: no eval rows")
        continue
    co = [r["label_orig"] == "correct" for r in sub]
    au_d = M.auroc([r["conf"] for r in sub], co)
    au_o = M.auroc([r["conf_orig"] for r in sub], co)
    e_d = M.ece([r["conf"] for r in sub], co)
    e_o = M.ece([r["conf_orig"] for r in sub], co)
    mad = sum(abs(r["conf"] - r["conf_orig"]) for r in sub) / len(sub)
    mean_d = sum(r["conf"] for r in sub) / len(sub)
    mean_o = sum(r["conf_orig"] for r in sub) / len(sub)
    print(f"# {pert}: n={len(sub)} | AUROC {au_o:.3f}->{au_d:.3f} "
          f"| ECE {e_o:.3f}->{e_d:.3f} | mean conf {mean_o:.3f}->{mean_d:.3f} "
          f"| mean|d|={mad:.3f}")
    # 分库明细
    dbs = sorted({r["db_id"] for r in sub})
    for db in dbs:
        s = [r for r in sub if r["db_id"] == db]
        c2 = [r["label_orig"] == "correct" for r in s]
        print(f"    {db:26s} n={len(s)} "
              f"AUROC {M.auroc([r['conf_orig'] for r in s], c2):.3f}"
              f"->{M.auroc([r['conf'] for r in s], c2):.3f} "
              f"mean {sum(r['conf_orig'] for r in s) / len(s):.3f}"
              f"->{sum(r['conf'] for r in s) / len(s):.3f}")
