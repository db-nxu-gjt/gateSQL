# -*- coding: utf-8 -*-
"""convert_auc_verdicts.py — 把 AUC 判官输出转成 GATE-SQL 统一置信格式。

Laya (B6/B7) 聚合公式复刻自 AUC src/laya_judge/eval_judge.py::gate_conf:
  conf = min( 5 个 noul 题型的 sigmoid(logit(noul)/T), q2_shortlist.confidence,
              q6_intent.confidence )
  T 取自 AUC temperatures_{tag}.json 的 buckets["noul(k=2)"]["T"]。

B5 (verdicts_b1_*): llm.verdict ∈ {allow, alarm} 二值 → conf 1.0/0.0
（二元信号，AUROC/AURC 解释力有限，论文正式版需 rejudge 成连续分，如实记录）。

输出: {qid, source, cand_idx, conf, method} jsonl（仅 correct/silent 行，与
AUC 评测协议一致——exec_fail 不参与判官评测）。

用法:
  python convert_auc_verdicts.py --tag b3 --method b6_laya_zeroshot --out <...>.jsonl
  python convert_auc_verdicts.py --tag b4 --method b7_laya_temp --out <...>.jsonl
  python convert_auc_verdicts.py --verdicts verdicts_b1_full.jsonl --b5 --method b5_judge --out <...>.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os

# Internal-questionnaire judge artifacts (not redistributable); place exports under
# data/external/ to rerun the B6/B7 conversion (released pools already contain the
# converted verdicts, bird300_ds_b6verdicts.jsonl).
AUC_DJ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "external", "data_judge")
AUC_LJ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "external", "laya_judge")
NOUL = ["q1_schema", "q3_compare", "q4_group", "q5_values", "q7_safety"]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="b3/b4（自动找 verdicts 与温度）")
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--b5", action="store_true", help="二值 verdict 转换")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    verd = args.verdicts or (f"{AUC_DJ}\\verdicts_{args.tag}.jsonl"
                             if not args.b5 else f"{AUC_DJ}\\verdicts_b1_full.jsonl")

    labels: dict[tuple, str] = {}
    with open(os.path.join(AUC_DJ, "labels.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels[(r["qid"], r["source"], r["cand_idx"])] = r["label"]

    T = None
    if not args.b5:
        tp = os.path.join(AUC_LJ, f"temperatures_{args.tag}.json")
        if not os.path.exists(tp):  # b3 零样本温度在早期文件 temperatures.json 中
            tp = os.path.join(AUC_LJ, "temperatures.json")
        with open(tp, encoding="utf-8") as f:
            T = json.load(f)["buckets"]["noul(k=2)"]["T"]
        print(f"[convert] tag={args.tag} T={T} ({os.path.basename(tp)})")

    n, n_skip = 0, 0
    with open(verd, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            k = (r["qid"], r["source"], r["cand_idx"])
            lab = labels.get(k)
            if lab not in ("correct", "silent"):
                n_skip += 1
                continue
            if args.b5:
                conf = 1.0 if r.get("llm", {}).get("verdict") == "allow" else 0.0
            else:
                a = r["answers"]["answers"]
                ps = [sigmoid(math.log(max(1e-6, a[q]["noul"])
                                    / max(1e-6, 1 - a[q]["noul"])) / T)
                      for q in NOUL]
                ps.append(a["q2_shortlist"]["confidence"])
                ps.append(a["q6_intent"]["confidence"])
                conf = min(ps)
            out.write(json.dumps({"qid": k[0], "source": k[1], "cand_idx": k[2],
                                  "conf": round(conf, 6),
                                  "method": args.method},
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"[convert] {n} 行 -> {args.out} (skip {n_skip} 非 correct/silent)")


if __name__ == "__main__":
    main()
