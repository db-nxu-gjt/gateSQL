# -*- coding: utf-8 -*-
"""run_b6_judge.py — A1: Laya 7 题问卷零样本判官（DeepSeek API 版）。

复刻 AUC laya_judge 协议（judge_schema.py / state_builder.py）：
  state = [QUESTION]/[SCHEMA](引用表紧凑 DDL)/[VALUES](过滤列 distinct 样本)
          /[EVIDENCE]/[SQL]（token 预算放宽：API 无 1024 ctx 限制）
  7 题 = q1_schema/q3_compare/q4_group/q5_values/q7_safety(noul,yes/no)
       + q2_shortlist/q6_intent(choice)。
  laya 原版单次前向 7 题 → API 单次调用回答全部 7 题（JSON），
  贴近原协议且大幅省调用数。

gate_conf（零样本 T=1，同 convert_auc_verdicts 的 temp=False 分支）：
  noul P(yes) = answer=="yes" ? conf : 1-conf
  gate_conf = min(5 个 noul P(yes), q2.conf, q6.conf)

用法:
  python run_b6_judge.py --pool ../../data/pools/bird300_ds_pool.jsonl \
      --out ../../data/pools/bird300_ds_b6.jsonl \
      --limit-qids 200 --source nucleus10 --workers 8
  python run_b6_judge.py --convert-only ...   # 仅从 verdicts 转 conf
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
from common.schema import load_pool, load_labels, write_jsonl  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402
from common.exec_match import db_path_of  # noqa: E402
import sqlglot
from sqlglot import exp  # noqa: E402

CACHE = os.path.join(_ROOT, "data", "ddl")

# ---- 7 题指令（逐字复刻 judge_schema.py）----
Q1 = ("Given the database schema provided in the state, answer: does every table "
      "and every column referenced in the SQL query exist in that schema? "
      "Answer yes only if all referenced tables and columns are present.")
Q3 = ("Read the natural language question and the SQL query. Are the comparison "
      "directions in WHERE/HAVING clauses (such as greater/less, at least/at most, "
      "before/after) consistent with what the question asks for? "
      "Answer yes only if every comparison direction matches the question semantics.")
Q4 = ("Does the aggregation granularity of the SQL query (GROUP BY columns and "
      "aggregate functions) match the granularity that the natural language "
      "question asks for? Answer yes only if the grouping level matches.")
Q5 = ("The state lists sample distinct values for some columns. For the literal "
      "values used in equality / IN / range filters of the SQL query, is each "
      "literal plausibly present among the actual column values? "
      "Answer yes only if no filter literal clearly contradicts the sampled values.")
Q7 = ("Check this SQL query for security risks: SQL injection payload patterns, "
      "destructive statements (DROP/DELETE/UPDATE without WHERE), or access to "
      "privileged system tables. Is the query free of all such risks? "
      "Answer yes only if it is safe.")
Q6_INTENTS = {
    "aggregation_sum": "computes a total/sum over rows",
    "aggregation_count": "counts rows or distinct values",
    "average_mean": "computes an average/mean value",
    "ranking_topk": "returns top-k / bottom-k rows by ordering",
    "comparison_extremum": "finds the maximum or minimum entity",
    "existence_check": "checks whether any matching row exists",
    "proportion_ratio": "computes a ratio, share, or percentage",
    "difference_delta": "computes a difference between two values or groups",
    "group_listing": "lists groups with aggregates per group",
    "temporal_recency": "selects the most recent / earliest records",
    "set_operation": "combines or contrasts sets (IN / UNION / EXCEPT)",
    "plain_lookup": "plain row lookup filtered by given attributes",
}
Q6 = ("Which intent best describes what this SQL query computes, given the "
      "natural language question? Pick exactly one intent label.")
Q2 = ("The SQL query references columns that may be misspelled or ambiguous. "
      "For the queried phrase, which real schema column from the candidate list "
      "does it most plausibly correspond to?")

QIDS = ["q1_schema", "q2_shortlist", "q3_compare", "q4_group",
        "q5_values", "q6_intent", "q7_safety"]
NOUL = ["q1_schema", "q3_compare", "q4_group", "q5_values", "q7_safety"]

SYSTEM = ("You are a precise SQL auditor. Answer every question about the "
          "candidate SQL strictly based on the provided state. "
          'Output ONLY a single JSON object mapping each question id to an '
          'object {"answer": <"yes" or "no" for yes/no questions; the chosen '
          'label for choice questions>, "confidence": <number 0.0-1.0, your '
          "certainty in the answer>}. "
          'Example: {"q1_schema": {"answer": "yes", "confidence": 0.9}, '
          '"q2_shortlist": {"answer": "table.column", "confidence": 0.7}, ...}'
          " No markdown, no extra text.")

_CONSTRAINT_RE = re.compile(
    r"^\s*(primary\s+key|foreign\s+key|unique|check|constraint)\b", re.I)


def _ddl_for_db(db_id: str) -> dict:
    fp = os.path.join(CACHE, f"ddl_{db_id}.json")
    if os.path.exists(fp):
        return json.load(open(fp, encoding="utf-8"))
    con = sqlite3.connect("file:%s?mode=ro" % db_path_of(db_id).replace("\\", "/"),
                          uri=True)
    rows = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    con.close()
    ddl = {name.lower(): sql for name, sql in rows}
    json.dump(ddl, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
    return ddl


def _compact_ddl(block: str) -> str:
    out = []
    for ln in block.splitlines():
        if _CONSTRAINT_RE.match(ln):
            continue
        s = ln.strip()
        if re.search(r"\b(primary\s+key|references|unique)\b", s, re.I):
            s = re.split(r"\b(?:primary\s+key|references|unique)\b", s, 1,
                         flags=re.I)[0].strip().rstrip(",")
        out.append(s if not out else "    " + s)
    return "\n".join(out)


def _filter_columns(sql: str):
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return []
    alias2tbl = {}
    for t in tree.find_all(exp.Table):
        real = (t.name or "").lower()
        al = (t.alias or "").lower()
        if real and al:
            alias2tbl[al] = real
    out = []
    for node in tree.find_all(exp.Binary, exp.In):
        col = None
        if isinstance(node, exp.Binary) and isinstance(node.this, exp.Column):
            col = node.this
        elif isinstance(node, exp.In) and isinstance(node.this, exp.Column):
            col = node.this
        if col is None:
            continue
        tbl = (col.table or "").lower()
        real = alias2tbl.get(tbl, tbl) or ""
        if not real:
            continue
        out.append(f"{real}.{col.name.lower()}")
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:8]


def _values_for(db_id: str, colspec: str, n=12):
    fp = os.path.join(CACHE, f"values_{db_id}.json")
    cache = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}
    key = f"{colspec}|{n}"
    if key not in cache:
        tbl, col = colspec.split(".", 1) if "." in colspec else ("", colspec)
        try:
            con = sqlite3.connect(
                "file:%s?mode=ro" % db_path_of(db_id).replace("\\", "/"), uri=True)
            q = ('SELECT DISTINCT "%s" FROM "%s" LIMIT %d' % (col, tbl, n)) \
                if tbl else ("SELECT DISTINCT \"%s\" FROM sqlite_master LIMIT %d"
                             % (col, n))
            rows = con.execute(q).fetchall()
            con.close()
            vals = [("" if v is None else str(v))[:24] for (v,) in rows]
        except Exception:
            vals = []
        cache[key] = vals
        json.dump(cache, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
    return cache[key]


def build_state(c) -> str:
    """AUC state_builder 协议（token 预算放宽至 ~3000，无 ctx 限制）。"""
    try:
        refs = sorted({t.name.lower() for t in
                       sqlglot.parse_one(c.sql, dialect="sqlite")
                       .find_all(exp.Table)})
    except Exception:
        refs = []
    ddl_map = _ddl_for_db(c.db_id)
    blocks = [_compact_ddl(ddl_map[t]) for t in refs if t in ddl_map]

    val_lines = []
    for colspec in _filter_columns(c.sql):
        vals = _values_for(c.db_id, colspec)
        if vals:
            val_lines.append(f"{colspec}: {' | '.join(vals[:12])}")
    values_txt = ("[VALUES]\n" + "\n".join(val_lines) + "\n") if val_lines else ""
    ev_txt = (f"[EVIDENCE] {c.evidence.strip()}\n") if c.evidence else ""
    head = f"[QUESTION] {c.question.strip()}\n"
    return (head + "[SCHEMA]\n" + "\n".join(blocks) + "\n"
            + values_txt + ev_txt + f"[SQL]\n{c.sql.strip()}\n")


def q2_criteria(c, ddl_map: dict) -> dict:
    """SQL 引用列所属表的全部列（API 版无检索索引，取全列消歧）。"""
    cols = set()
    for colspec in _filter_columns(c.sql):
        tbl = colspec.split(".")[0]
        block = ddl_map.get(tbl, "")
        for m in re.finditer(r'["\[]?(\w+)["\]]?\s+\w+', block):
            cols.add(f"{tbl}.{m.group(1).lower()}")
    return {cc: "candidate column" for cc in sorted(cols)[:20]}


def build_prompt(state: str, crit2: dict) -> str:
    parts = [
        f"q1_schema: {Q1}",
        f"q2_shortlist: {Q2} Choose exactly one label from: "
        + json.dumps(list(crit2), ensure_ascii=False),
        f"q3_compare: {Q3}",
        f"q4_group: {Q4}",
        f"q5_values: {Q5}",
        f"q6_intent: {Q6} Choose exactly one label from: "
        + json.dumps(list(Q6_INTENTS), ensure_ascii=False),
        f"q7_safety: {Q7}",
    ]
    return ("[STATE]\n" + state + "\n\n[QUESTIONS]\n" + "\n".join(parts)
            + "\n\nRespond with ONLY the JSON object with keys: "
            + ", ".join(QIDS) + ".")


def judge_one(cli: LLMClient, c, ddl_map: dict, max_retry_done: set) -> dict | None:
    state = build_state(c)
    crit2 = q2_criteria(c, ddl_map)
    prompt = build_prompt(state, crit2)
    for _ in range(3):
        try:
            txt = cli.generate(prompt, n=1, temperature=0.0, max_tokens=600,
                               system=SYSTEM)[0]
            txt = txt.strip()
            txt = txt[txt.index("{"): txt.rindex("}") + 1]
            a = json.loads(txt)
            out = {}
            ok = True
            for q in QIDS:
                item = a.get(q, "")
                if isinstance(item, dict):
                    ans = str(item.get("answer", item.get("choice", ""))).lower()
                    conf = item.get("confidence")
                else:  # 扁平格式：值直接是答案字符串
                    ans = str(item).lower()
                    conf = None
                conf = float(conf) if conf is not None else None
                conf = min(max(conf, 0.0), 1.0) if conf is not None else None
                if q in NOUL:
                    if ans not in ("yes", "no"):
                        ok = False
                        break
                    if conf is None:  # 二值回退
                        conf = 1.0 if ans == "yes" else 0.0
                else:  # choice 题
                    if not ans:
                        ok = False
                        break
                    if conf is None:
                        conf = 1.0 if ans in (crit2 if q == "q2_shortlist"
                                              else Q6_INTENTS) else 0.5
                out[q] = {"answer": ans, "confidence": conf}
            if ok:
                return out
            time.sleep(1)
        except Exception:
            time.sleep(2)
    return None


def gate_conf_zero(answers: dict) -> float:
    ps = []
    for q in NOUL:
        a = answers[q]
        p = a["confidence"] if a["answer"] == "yes" else 1.0 - a["confidence"]
        ps.append(p)
    ps.append(answers["q2_shortlist"]["confidence"])
    ps.append(answers["q6_intent"]["confidence"])
    return min(ps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True, help="verdicts 输出 jsonl")
    ap.add_argument("--conf-out", default=None, help="gate_conf 置信文件输出")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--limit-qids", type=int, default=0,
                    help="只评 qid < N（子集验证）")
    ap.add_argument("--source", default="nucleus10")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--convert-only", action="store_true")
    args = ap.parse_args()

    all_qids = sorted({c.qid for c in load_pool(args.pool)})
    sub = set(all_qids[: args.limit_qids]) if args.limit_qids > 0 else None
    pool = [c for c in load_pool(args.pool)
            if c.source == args.source
            and (sub is None or c.qid in sub)]
    print(f"[b6] 候选 {len(pool)} (source={args.source}, "
          f"前 {args.limit_qids or 'all'} 题)")

    if not args.convert_only:
        done = set()
        if os.path.exists(args.out):
            for line in open(args.out, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    done.add((r["qid"], r["source"], r["cand_idx"]))
        todo = [c for c in pool if c.key not in done]
        print(f"[b6] done={len(done)} todo={len(todo)}")
        cli = LLMClient()
        ddl_cache = {}
        fp = open(args.out, "a", encoding="utf-8")
        n_err = 0

        def work(c):
            nonlocal n_err
            if c.db_id not in ddl_cache:
                ddl_cache[c.db_id] = _ddl_for_db(c.db_id)
            ans = judge_one(cli, c, ddl_cache[c.db_id], done)
            return c, ans

        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for i, (c, ans) in enumerate(ex.map(work, todo)):
                    if ans is None:
                        n_err += 1
                        continue
                    fp.write(json.dumps({
                        "qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                        "db_id": c.db_id, "model": "deepseek-b6-zero",
                        "answers": ans}, ensure_ascii=False) + "\n")
                    if (i + 1) % 100 == 0:
                        fp.flush()
                        dt = time.time() - t0
                        print(f"  {i + 1}/{len(todo)} {dt / max(1, i + 1):.2f}s/条 "
                              f"err={n_err}")
        finally:
            fp.close()
        print(f"[b6] 完成 todo={len(todo)} err={n_err}")

    if args.conf_out:
        verd = {}
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                verd[(r["qid"], r["source"], r["cand_idx"])] = r["answers"]
        rows = []
        for c in pool:
            if c.key not in verd:
                continue
            rows.append({"qid": c.qid, "source": c.source,
                         "cand_idx": c.cand_idx,
                         "conf": round(gate_conf_zero(verd[c.key]), 6),
                         "method": "b6_laya_zero_ds"})
        write_jsonl(args.conf_out, rows)
        print(f"[b6] conf {len(rows)} -> {args.conf_out}")


if __name__ == "__main__":
    main()
