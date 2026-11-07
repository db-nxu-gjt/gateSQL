# -*- coding: utf-8 -*-
"""B3a SAC-Avg — SQL-syntax-aware token 置信聚合（白盒，AAAI 2026 最优）。

出处: Maleki, Pourreza & Rafiei, AAAI 2026 (arXiv:2508.14056) 的 SAC：
  1. SQL-syntax-aware folding：把构成同一标识符/关键词的连续 subword
     token 折叠为一个单元（单元概率 = 其 token 概率的几何平均）；
  2. 排除非关键 token（结构关键词、标点、字面量、数字）；
  3. SAC-Avg = 关键单元概率的算术平均。

输入: 带 token_logprobs 的开源模型候选池（make_pool.py --logprobs 生成）
输出: confidence jsonl

用法:
  python run_b3_sac.py --pool PATH --out PATH [--mode avg]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import load_pool, write_jsonl  # noqa: E402

# 非关键 token（结构关键词 / 标点 / 空白 / 字面量），SAC 协议排除
NON_CRITICAL = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "LIMIT", "JOIN",
    "INNER", "LEFT", "RIGHT", "OUTER", "ON", "AND", "OR", "AS", "ASC",
    "DESC", "HAVING", "DISTINCT", "IN", "NOT", "NULL", "IS", "LIKE",
    "BETWEEN", "UNION", "ALL", "ANY", "EXISTS", "CASE", "WHEN", "THEN",
    "ELSE", "END", "INTERSECT", "EXCEPT",
}
AGG = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _strip_tok(t: str) -> str:
    """BPE token 去掉 Ġ/▁ 前缀。"""
    return t.replace("Ġ", " ").replace("▁", " ").strip()


def fold_tokens(tokens: list[str], lps: list[float]) -> list[tuple[str, float]]:
    """把连续 token 折叠成 word 单元，单元 logprob = token logprob 均值。"""
    units: list[list[int]] = []
    for i, t in enumerate(tokens):
        s = _strip_tok(t)
        if not units:
            units.append([i])
            continue
        prev = _strip_tok(tokens[units[-1][-1]])
        # 前一 token 未结束词（无尾边界）且当前 token 是续接（无空格开头）
        prev_ends = prev.endswith((" ", ",", "(", ")", ";", "'"))
        cont = (not s.startswith(" ")) and (not prev_ends) and s != ""
        if cont and not prev.endswith(";"):
            units[-1].append(i)
        else:
            units.append([i])
    out = []
    for idxs in units:
        text = "".join(_strip_tok(tokens[i]) for i in idxs)
        lp = sum(lps[i] for i in idxs) / len(idxs)  # 几何平均
        out.append((text, lp))
    return out


def critical_units(units: list[tuple[str, float]]) -> list[float]:
    """过滤出关键单元，返回其 logprob。"""
    keep = []
    for text, lp in units:
        u = text.strip()
        if not u or u.startswith("'") or u.endswith("'"):
            continue  # 字面量
        if not WORD_RE.search(u):
            continue  # 纯标点/空白
        up = u.upper()
        if up in NON_CRITICAL or up in AGG:
            continue  # 结构关键词与聚合名
        if re.fullmatch(r"[-+*/%(),.;=<>!?]+", u):
            continue
        keep.append(lp)
    return keep


def sac_avg(tokens: list[str], lps: list[float]) -> float:
    units = fold_tokens(tokens, lps)
    crit = critical_units(units)
    if not crit:  # 全被过滤（退化），退回全体 token 平均
        crit = [lp for _, lp in units] or lps
    return math.exp(sum(crit) / len(crit))


def sac_prod(tokens: list[str], lps: list[float]) -> float:
    """SAC-Product：关键单元 logprob 求和后 exp（供消融）。"""
    units = fold_tokens(tokens, lps)
    crit = critical_units(units)
    if not crit:
        crit = [lp for _, lp in units] or lps
    return math.exp(sum(crit))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="带 token_logprobs 的候选池")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="avg", choices=["avg", "product"])
    args = ap.parse_args()

    pool = load_pool(args.pool, require_logprobs=True)
    fn = sac_avg if args.mode == "avg" else sac_prod
    rows = []
    for c in pool:
        conf = fn(c.tokens, c.token_logprobs)
        conf = min(1.0, max(1e-9, conf))
        rows.append({"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                     "conf": conf, "method": f"b3_sac_{args.mode}"})
    write_jsonl(args.out, rows)
    print(f"[done] {len(rows)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
