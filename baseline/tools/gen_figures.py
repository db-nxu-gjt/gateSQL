# -*- coding: utf-8 -*-
"""gen_figures.py — 论文图表（英文，PDF 矢量，Okabe-Ito 色板）。

输出 paper/submit/main/figures/fig_*.pdf + .png(300dpi)。
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
from common import metrics as M  # noqa: E402

OUT = os.path.normpath(os.path.join(_ROOT, "paper", "submit", "main",
                                    "figures"))
os.makedirs(OUT, exist_ok=True)
POOLDIR = os.path.join(_ROOT, "data", "pools")

# Okabe-Ito (colorblind safe)
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


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)
    print("fig:", name)


def main() -> None:
    labels = load_labels_raw()
    confs = {n: load_conf(f) for n, f in (
        ("B1", "bird300_ds_b1.jsonl"),
        ("B4", "bird300_ds_b4.jsonl"),
        ("Fuse-min", "bird300_ds_fuse_min.jsonl"),
        ("Fuse-prod", "bird300_ds_fuse_prod.jsonl"),
        ("Fuse-stack", "bird300_ds_fuse_stack.jsonl"),
        ("B3", "bird300_ds_b3_sac.jsonl"))}
    keysA = None
    for n in ("B1", "B3", "B4", "Fuse-prod"):
        ks = set(confs[n]) & set(labels)
        keysA = ks if keysA is None else keysA & ks
    keysA = sorted(keysA)

    # ---------- Fig 1: risk-coverage ----------
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    covs = np.linspace(0.05, 1.0, 20)
    ex0 = sum(labels[k] == "correct" for k in keysA) / len(keysA)
    for (n, c, mk) in (("B1", OI[0], "o"), ("B4", OI[3], "s"),
                       ("Fuse-min", OI[1], "^"),
                       ("Fuse-prod", OI[2], "D"),
                       ("Fuse-stack", OI[4], "v")):
        ys = []
        for cov in covs:
            ranked = sorted(keysA, key=lambda k: -confs[n][k])
            take = ranked[: max(1, int(len(ranked) * cov))]
            ys.append(sum(labels[k] == "correct" for k in take) / len(take))
        ax.plot(covs * 100, ys, color=c, marker=mk, ms=3, lw=1.4,
                markevery=3, label=n)
    ax.axhline(ex0, color="k", ls="--", lw=1, label="Always-execute")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Execution accuracy")
    ax.set_title("Risk\u2013coverage on BIRD-300 (DeepSeek pool)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "fig_risk_coverage")

    # ---------- Fig 2: reliability diagram ----------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (n, c) in zip(axes, (("B1", OI[0]), ("B4", OI[3]),
                                 ("Fuse-prod", OI[2]))):
        co = np.array([labels[k] == "correct" for k in keysA], float)
        sc = np.array([confs[n][k] for k in keysA])
        bins = [[] for _ in range(10)]
        for s, y in zip(sc, co):
            bins[min(9, int(s * 10))].append((s, y))
        xs, ys, ns = [], [], []
        for b in bins:
            if b:
                xs.append(np.mean([p for p, _ in b]))
                ys.append(np.mean([y for _, y in b]))
                ns.append(len(b))
        ax.bar(xs, ys, width=0.09, color=c, alpha=0.75,
               edgecolor="white", lw=0.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{n}  (ECE {M.ece(list(sc), list(co)):.3f})")
        ax.set_xlabel("Predicted confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Empirical accuracy")
    axes[0].text(0.06, 0.9, "ideal", transform=axes[0].transAxes,
                 fontsize=7, rotation=37)
    save(fig, "fig_reliability")

    # ---------- Fig 3: LODO grouped bars ----------
    lodo = json.load(open(os.path.join(_ROOT, "results",
                                       "_lodo_rows.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(_ROOT, "results",
                                       "_lodo_rows.json")) else None
    if lodo is None:
        # 直接重算（同 gen_excel 逻辑），此处读取由 gen_excel 落盘的中转文件
        raise SystemExit("run gen_excel.py first (writes _lodo_rows.json)")
    dbs = [r[0] for r in lodo[:-1]]
    b4 = [r[4] for r in lodo[:-1]]
    fs = [r[5] for r in lodo[:-1]]
    m4, mf = lodo[-1][4], lodo[-1][5]
    x = np.arange(len(dbs))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(x - 0.19, b4, 0.38, color=OI[3], label="B4-LODO")
    ax.bar(x + 0.19, fs, 0.38, color=OI[2], label="Fuse-LODO")
    ax.axhline(m4, color=OI[3], ls=":", lw=1.2)
    ax.axhline(mf, color=OI[2], ls=":", lw=1.2)
    ax.text(len(dbs) - 0.4, m4 + 0.01, "macro B4", fontsize=7,
            color=OI[3], ha="right")
    ax.text(len(dbs) - 0.4, mf + 0.01, "macro Fuse", fontsize=7,
            color=OI[2], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dbs], fontsize=6.5)
    ax.set_ylabel("AUROC (held-out DB)")
    lo = min(b4 + fs + [m4, mf])
    ax.set_ylim(max(0.3, min(0.4, lo - 0.05)), 0.95)
    ax.set_title("Leave-one-database-out generalization (11 folds)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save(fig, "fig_lodo")

    # ---------- Fig 4: drift ----------
    dr = json.load(open(os.path.join(_ROOT, "results",
                                     "_drift_rows.json"), encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    kinds = [("rename", OI[0], "o"), ("add_col", OI[2], "s"),
             ("split_tbl", OI[1], "D")]
    for pert, c, mk in kinds:
        rows = [r for r in dr if r[0].endswith("/" + pert)]
        ax.scatter([r[1] for r in rows], [r[2] for r in rows], color=c,
                   marker=mk, s=28, label=pert)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("Candidates per variant")
    ax.set_ylabel("Label agreement")
    ax.set_ylim(0.8, 1.02)
    ax.set_title("(a) Label agreement under drift")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    for pert, c, mk in kinds:
        rows = [r for r in dr if r[0].endswith("/" + pert)]
        ax.scatter([r[4] for r in rows], [r[3] for r in rows], color=c,
                   marker=mk, s=28, label=pert)
    ax.plot([0.4, 0.9], [0.4, 0.9], "k--", lw=1)
    ax.set_xlabel("AUROC on original pool")
    ax.set_ylabel("AUROC under drift (B4)")
    ax.set_title("(b) B4 ranking: drifted vs. original")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_drift")

    # ---------- Fig 5: conformal 20 seeds ----------
    cr = json.load(open(os.path.join(_ROOT, "results",
                                     "_conf_rows.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    marks = [("B4", OI[3], "s"), ("Fuse-prod", OI[2], "D")]
    for wi, (name, c, mk) in enumerate(marks):
        row = next(r for r in cr if r[0] == name)
        seeds = list(range(20))
        leaks = [p[1] for p in row[8]]
        covs = [p[3] for p in row[8]]
        ax.scatter([s + wi * 0.35 - 0.17 for s in seeds], leaks, color=c,
                   marker=mk, s=22, label=name, alpha=0.9)
        for s, cv, lk in zip(seeds, covs, leaks):
            ax.plot([s + wi * 0.35 - 0.17] * 2, [0, cv], color=c,
                    lw=0.5, alpha=0.25)
    ax.axhline(0.02, color="k", ls="--", lw=1.2)
    ax.text(19.4, 0.024, r"$\alpha=0.02$", fontsize=8, ha="right")
    ax.set_xlabel("Calibration seed")
    ax.set_ylabel(r"$P(\mathrm{accept}\mid\mathrm{wrong})$")
    ax.set_title("Conformal guarantee violation (20 seeds)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_conformal")

    # ---------- Fig 6: difficulty ----------
    dev = json.load(open(os.path.join(_ROOT, "data", "raw", "bird",
                                      "dev.json"), encoding="utf-8"))
    diff_of = {i: it.get("difficulty", "") for i, it in enumerate(dev)}
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ds = ["simple", "moderate", "challenging"]
    methods = [("B1", OI[0]), ("B4", OI[3]), ("Fuse-prod", OI[2])]
    w = 0.25
    for mi, (n, c) in enumerate(methods):
        ys = []
        for d in ds:
            keys = [k for k in keysA if diff_of.get(k[0]) == d]
            co = [labels[k] == "correct" for k in keys]
            ys.append(M.auroc([confs[n][k] for k in keys], co))
        ax.bar(np.arange(3) + (mi - 1) * w, ys, w, color=c, label=n)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Simple", "Moderate", "Challenging"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 0.7)
    ax.set_title("AUROC by question difficulty")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save(fig, "fig_difficulty")

    # ---------- Fig 7: laya triangle ----------
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    names = ["Local base\n(AUC pool)", "Local FT+temp\n(AUC pool)",
             "DeepSeek zero-shot\n(our pool)"]
    vals = [0.5605, 0.5032, 0.537]
    bars = ax.bar(names, vals, 0.55, color=[OI[5], OI[1], OI[0]])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.text(2.42, 0.505, "random", fontsize=7, ha="right")
    ax.axhline(0.666, color=OI[2], ls=":", lw=1.4)
    ax.text(2.42, 0.672, "Fuse-prod", fontsize=7, color=OI[2], ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 0.72)
    ax.set_title("7-question judge protocol across executors")
    ax.tick_params(axis="x", labelsize=7)
    save(fig, "fig_laya")


if __name__ == "__main__":
    main()
