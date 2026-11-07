# -*- coding: utf-8 -*-
"""gen_fig_lodo_matched.py — fig_lodo: matched-protocol LODO (global vs refit).

Schema of _lodo_matched_rows.json rows (folds then MACRO/POOLED):
  [db, n, b1, b4glb, b4lo, fuseglb, fuselo,
   e_b4glb, e_b4lo, e_fuseglb, e_fuselo]
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.normpath(os.path.join(_ROOT, "figures"))

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

rows = json.load(open(os.path.join(_ROOT, "results",
                                   "_lodo_matched_rows.json"),
                      encoding="utf-8"))
folds = [r for r in rows if r[0] not in ("MACRO (11 folds)", "POOLED (all DBs)")]
macro = next(r for r in rows if r[0] == "MACRO (11 folds)")

dbs = [r[0] for r in folds]
b4g = [r[3] for r in folds]
b4l = [r[4] for r in folds]
fsg = [r[5] for r in folds]
fsl = [r[6] for r in folds]
m4l, mfl = macro[4], macro[6]

x = np.arange(len(dbs))
w = 0.19
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.bar(x - 1.5 * w, b4g, w, color=OI[3], alpha=0.40, label="B4 global")
ax.bar(x - 0.5 * w, b4l, w, color=OI[3], label="B4 LODO")
ax.bar(x + 0.5 * w, fsg, w, color=OI[2], alpha=0.40, label="Fuse global")
ax.bar(x + 1.5 * w, fsl, w, color=OI[2], label="Fuse LODO")
ax.axhline(m4l, color=OI[3], ls=":", lw=1.2)
ax.axhline(mfl, color=OI[2], ls=":", lw=1.2)
ax.text(len(dbs) - 0.4, m4l - 0.035, "macro B4 LODO", fontsize=7,
        color=OI[3], ha="right")
ax.text(len(dbs) - 0.4, mfl + 0.012, "macro Fuse LODO", fontsize=7,
        color=OI[2], ha="right")
ax.set_xticks(x)
ax.set_xticklabels([d.replace("_", "\n") for d in dbs], fontsize=6.5)
ax.set_ylabel("AUROC (per-database, matched eval pool)")
lo = min(b4g + b4l + fsg + fsl)
ax.set_ylim(max(0.1, lo - 0.06), 0.98)
ax.set_title("Leave-one-database-out: global model vs.\\ LODO refit "
             "(identical eval candidates per fold)")
ax.legend(frameon=False, loc="upper left", ncol=4, columnspacing=1.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_lodo.{ext}"), dpi=300)
print("saved fig_lodo.{pdf,png} ->", OUT)
