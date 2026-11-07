# -*- coding: utf-8 -*-
"""make_summary.py — Phase 5 汇总：统一口径主表 + risk-coverage + 按难度分解。

口径（与 Phase 2/3 align_eval 一致）：
  - AUROC/ECE15 在 correct+silent 上计算
  - sel_ex@cov 在全集（含 exec_fail）上计算
  - 口径 A：白盒交集（b1/b3/b4/fuse* 全有 conf 的键）
  - 口径 B：全方法交集（再 ∩ b2/b5）

用法:
  python make_summary.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
from common import metrics as M  # noqa: E402

POOLDIR = os.path.join(_ROOT, "data", "pools")
BIRD_DEV = os.path.join(_ROOT, "data", "raw", "bird", "dev.json")

METHODS = {
    "B1 自报置信": "bird300_ds_b1.jsonl",
    "B2 一致性(k=10)": "bird300_ds_b2_nuc10.jsonl",
    "B3 SAC-Avg": "bird300_ds_b3_sac.jsonl",
    "B4 重标定": "bird300_ds_b4.jsonl",
    "B5 判官连续": "bird300_ds_b5.jsonl",
    "fuse_min": "bird300_ds_fuse_min.jsonl",
    "fuse_prod": "bird300_ds_fuse_prod.jsonl",
    "fuse_stack": "bird300_ds_fuse_stack.jsonl",
}


def load_conf(fn: str) -> dict:
    out = {}
    with open(os.path.join(POOLDIR, fn), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    float(r["conf"])
    return out


def main() -> None:
    labels = {}
    with open(os.path.join(POOLDIR, "bird300_ds_labels.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = \
                    r["label"]

    confs = {name: load_conf(fn) for name, fn in METHODS.items()}
    keys_A = None
    for name in ("B1 自报置信", "B3 SAC-Avg", "B4 重标定", "fuse_prod"):
        ks = set(confs[name]) & set(labels)
        keys_A = ks if keys_A is None else keys_A & ks
    keys_B = set(keys_A)
    for name in ("B2 一致性(k=10)", "B5 判官连续"):
        keys_B &= set(confs[name]) & set(labels)

    def eval_keys(keys, name):
        co = [labels[k] == "correct" for k in keys]
        sc = [confs[name][k] for k in keys]
        au = M.auroc(sc, co)
        ece = M.ece(sc, co)
        pairs = sorted(zip(sc, [labels[k] for k in keys]),
                       key=lambda x: -x[0])
        # sel_ex@cov 全集口径（含 exec_fail）：keys 需含 exec_fail
        return au, ece

    print("## 口径 A（白盒交集 n=%d, keys 含 exec_fail=%d）" % (
        len(keys_A), sum(labels[k] == "exec_fail" for k in keys_A)))
    print("| 方法 | AUROC | ECE15 |")
    print("|---|---|---|")
    for name in METHODS:
        if name in ("B2 一致性(k=10)", "B5 判官连续"):
            continue
        au, ece = eval_keys(keys_A, name)
        print("| %s | %.3f | %.3f |" % (name, au, ece))
    n_corr = sum(labels[k] == "correct" for k in keys_A)
    print("| B0 总是执行 | — | — |  (EX=%.3f)" % (n_corr / len(keys_A)))

    print("\n## 口径 B（全方法交集 n=%d）" % len(keys_B))
    print("| 方法 | AUROC | ECE15 |")
    print("|---|---|---|")
    for name in METHODS:
        if name in ("B2 一致性(k=10)", "B5 判官连续"):
            keys = keys_B
        else:
            keys = keys_B
        au, ece = eval_keys(keys, name)
        print("| %s | %.3f | %.3f |" % (name, au, ece))

    # risk-coverage 关键点（口径 A，全集含 exec_fail）
    print("\n## risk-coverage（口径 A 全集 n=%d）" % len(keys_A))
    print("| cov | " + " | ".join(
        n for n in ("B1 自报置信", "B4 重标定", "fuse_min", "fuse_prod")) + " |")
    print("|---|---|---|---|---|")
    full = {n: confs[n] for n in ("B1 自报置信", "B4 重标定", "fuse_min",
                                  "fuse_prod")}
    for cov in (0.3, 0.5, 0.7, 0.9):
        row = []
        for n, cf in full.items():
            ranked = sorted(keys_A, key=lambda k: -cf[k])
            take = ranked[: int(len(ranked) * cov)]
            ex = sum(labels[k] == "correct" for k in take) / max(1, len(take))
            row.append(ex)
        print("| %.1f | " % cov + " | ".join("%.3f" % x for x in row) + " |")

    # 按难度分解（BIRD dev difficulty，qid=行号）
    dev = json.load(open(BIRD_DEV, encoding="utf-8"))
    diff_of = {i: it.get("difficulty", "") for i, it in enumerate(dev)}
    print("\n## 按难度分解（口径 A，AUROC / sel_ex@50 全集）")
    print("| 难度 | n | B1 | B4 | fuse_prod |")
    print("|---|---|---|---|---|")
    for d in ("simple", "moderate", "challenging"):
        keys = [k for k in keys_A if diff_of.get(k[0]) == d]
        if not keys:
            continue
        cells = []
        for n in ("B1 自报置信", "B4 重标定", "fuse_prod"):
            cf = confs[n]
            co = [labels[k] == "correct" for k in keys]
            au = M.auroc([cf[k] for k in keys], co)
            ranked = sorted(keys, key=lambda k: -cf[k])
            take = ranked[: len(ranked) // 2]
            ex = sum(labels[k] == "correct" for k in take) / max(1, len(take))
            cells.append("%.3f/%.3f" % (au, ex))
        n_corr = sum(labels[k] == "correct" for k in keys)
        print("| %s | %d (EX %.3f) | %s |" % (d, len(keys),
                                              n_corr / len(keys),
                                              " | ".join(cells)))


if __name__ == "__main__":
    main()
