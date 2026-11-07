# -*- coding: utf-8 -*-
"""run_b1_variants.py — B1 改进 elicitation 变体（meta-review B2(b) 修订项）。

两个变体（文献对齐: tian2023just / xiong2024can）:
  2stage : multi-turn elicitation —— 先让模型自己作答（Stage 1, 每题 1 次），
           再对每个候选给 0-100 置信（Stage 2, 每候选 1 次）。
  topk5  : top-k elicitation —— 每题 1 次输出 top-5 最可能 SQL 及概率，
           候选置信 = 其在 top-5 中的概率（未列入 = 0）。

只跑 seed-0 eval 切分（与正文口径一致；training-free 方法不使用 dev 段）。
输出 conf jsonl（method=b1_2stage / b1_topk5），断点续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
from common.schema import load_pool, iter_jsonl  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402

POOLS = os.path.join(_ROOT, "data", "pools")
LAB_F = os.path.join(POOLS, "bird300_ds_labels.jsonl")

STAGE1_PROMPT = """You are a Text-to-SQL expert. Given the database schema, question and hint, write ONE SQLite query.

### Schema
{schema}

### Question
{question}
{evidence}
Respond with ONLY the SQL query (single line, no markdown, no explanation)."""

STAGE2_PROMPT = """You are given a database schema, a natural language question, your own proposed SQL query, and a candidate SQL query.

### Schema
{schema}

### Question
{question}
{evidence}
### Your proposed SQL
{own}

### Candidate SQL
{sql}

How confident are you that this candidate SQL query is correct (returns exactly the right answer)? Carefully compare it against the schema, the question, and your own proposed query.
Respond with ONLY a number between 0 and 100 (your confidence), nothing else."""

TOPK_PROMPT = """You are a Text-to-SQL expert. Given the database schema, question and hint, list the {k} most likely correct SQLite queries.

### Schema
{schema}

### Question
{question}
{evidence}
Output EXACTLY {k} lines, one per query, in this format (no markdown, no extra text):
<probability> | <single-line SQL query>
where <probability> is a decimal between 0 and 1, the probabilities sum to 1, ordered from most to least likely."""


def load_ddl(ddl_dir: str, db_id: str) -> str:
    path = os.path.join(ddl_dir, f"ddl_{db_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            tables = json.load(f)
        return "\n\n".join(tables.values())
    return ""


def parse_conf(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return 0.5
    v = float(m.group(1))
    if v > 1.0:
        v /= 100.0
    return min(1.0, max(0.0, v))


def norm_sql(s: str) -> str:
    return " ".join(s.strip().rstrip(";").lower().split())


def parse_topk(text: str, k: int) -> list:
    out = []
    for line in text.strip().splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        m = re.match(r"^([01]?(?:\.\d+)?|[01])\s*[|]\s*(.+)$", line)
        if not m:
            continue
        p = float(m.group(1))
        if p > 1.0:
            p /= 100.0
        out.append((p, norm_sql(m.group(2))))
    if out and sum(p for p, _ in out) > 1.05:
        s = sum(p for p, _ in out)
        out = [(p / s, sql) for p, sql in out]
    return out[:k]


def eval_qids() -> set:
    labs = {int(json.loads(l)["qid"]) for l in open(LAB_F, encoding="utf-8")
            if l.strip()}
    qids = sorted(labs)
    rng = random.Random(0)
    rng.shuffle(qids)
    n_fit = max(1, int(len(qids) * 0.3))
    return set(qids[n_fit:])


def merge(out_path: str, rows: list) -> list:
    old = []
    if os.path.exists(out_path):
        old = [json.loads(l) for l in open(out_path, encoding="utf-8")
               if l.strip()]
    seen = {(r["qid"], r["source"], r["cand_idx"]) for r in old}
    return old + [r for r in rows
                  if (r["qid"], r["source"], r["cand_idx"]) not in seen]


def write_merge(out_path: str, rows: list) -> None:
    if not rows:
        return
    merged = merge(out_path, rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["2stage", "topk5"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ddl-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ddl"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ev = eval_qids()
    print(f"[split] seed-0 eval: {len(ev)} questions")
    pool = [c for c in load_pool(os.path.join(POOLS, "bird300_ds_pool.jsonl"))
            if c.qid in ev]
    by_q = {}
    for c in pool:
        by_q.setdefault(c.qid, []).append(c)
    qids = sorted(by_q)
    print(f"[pool] eval candidates {len(pool)} / {len(qids)} questions")

    ddl_cache = {}
    cli = LLMClient(model=args.model)
    method = "b1_" + args.variant
    n_parse_fail = 0
    st1_cache_path = os.path.join(POOLS, "bird300_ds_b1_2stage_stage1.json")
    st1 = {}
    if args.variant == "2stage" and os.path.exists(st1_cache_path):
        st1 = json.load(open(st1_cache_path, encoding="utf-8"))
        print(f"[resume] stage1 has {len(st1)} questions")

    def schema_of(c) -> str:
        if c.db_id not in ddl_cache:
            ddl_cache[c.db_id] = load_ddl(args.ddl_dir, c.db_id)
        return ddl_cache[c.db_id]

    if args.variant == "topk5":
        k = 5
        done_q = set()
        if os.path.exists(args.out):
            done_q = {r["qid"] for r in iter_jsonl(args.out)}
        todo_q = [q for q in qids if q not in done_q]
        print(f"[topk5] todo {len(todo_q)} questions")

        def one_topk(q: int) -> dict:
            c0 = by_q[q][0]
            sch = schema_of(c0)
            evid = f"Hint: {c0.evidence}\n" if c0.evidence else ""
            prompt = TOPK_PROMPT.format(k=k, schema=sch,
                                        question=c0.question, evidence=evid)
            try:
                text = cli.generate(prompt, n=1, temperature=0.0,
                                    max_tokens=400)[0]
                top = parse_topk(text, k)
            except Exception as e:
                print(f"[warn] qid={q} {e}", flush=True)
                top = []
            probs = {sql: p for p, sql in top}
            rows = []
            for c in by_q[q]:
                p = probs.get(norm_sql(c.sql), 0.0)
                rows.append({"qid": q, "source": c.source,
                             "cand_idx": c.cand_idx, "conf": float(p),
                             "method": method})
            return {"rows": rows, "n_listed": len(top), "qid": q}

        pending = []
        n_done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex2:
            futs = [ex2.submit(one_topk, q) for q in todo_q]
            for fut in as_completed(futs):
                r = fut.result()
                pending.extend(r["rows"])
                n_parse_fail += (0 if r["n_listed"] else 1)
                n_done += 1
                if n_done % 50 == 0 or n_done == len(todo_q):
                    write_merge(args.out, pending)
                    pending = []
                    print(f"[topk5] {n_done}/{len(todo_q)}", flush=True)
        write_merge(args.out, pending)
        print(f"[done] parse_fail_questions={n_parse_fail} -> {args.out}")
        return

    todo_s1 = [q for q in qids if q not in st1]
    print(f"[2stage] stage1 todo {len(todo_s1)} questions")

    def one_s1(q: int):
        c0 = by_q[q][0]
        sch = schema_of(c0)
        evid = f"Hint: {c0.evidence}\n" if c0.evidence else ""
        prompt = STAGE1_PROMPT.format(schema=sch, question=c0.question,
                                      evidence=evid)
        try:
            t = cli.generate(prompt, n=1, temperature=0.0, max_tokens=384,
                             stop=["\n\n"])[0].strip()
        except Exception as e:
            print(f"[warn] stage1 qid={q} {e}", flush=True)
            t = ""
        return q, t

    with ThreadPoolExecutor(max_workers=args.workers) as ex1:
        for fut in as_completed([ex1.submit(one_s1, q) for q in todo_s1]):
            q, t = fut.result()
            st1[str(q)] = t
            if len(st1) % 50 == 0:
                json.dump(st1, open(st1_cache_path, "w", encoding="utf-8"))
                print(f"[2stage] stage1 {len(st1)}/{len(qids)}", flush=True)
    json.dump(st1, open(st1_cache_path, "w", encoding="utf-8"))
    print(f"[2stage] stage1 done {len(st1)} questions")

    done = set()
    if os.path.exists(args.out):
        done = {(r["qid"], r["source"], r["cand_idx"])
                for r in iter_jsonl(args.out)}
    todo = [c for c in pool if (c.qid, c.source, c.cand_idx) not in done]
    print(f"[2stage] stage2 todo {len(todo)} candidates")

    def one_s2(c) -> dict:
        sch = schema_of(c)
        evid = f"Hint: {c.evidence}\n" if c.evidence else ""
        own = st1.get(str(c.qid), "")
        own_block = own if own else "(no answer produced)"
        prompt = STAGE2_PROMPT.format(schema=sch, question=c.question,
                                      evidence=evid, own=own_block, sql=c.sql)
        try:
            text = cli.generate(prompt, n=1, temperature=0.0, max_tokens=16)[0]
            conf = parse_conf(text)
        except Exception as e:
            print(f"[warn] qid={c.qid} {e}", flush=True)
            conf = 0.5
        return {"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                "conf": conf, "method": method}

    pending2 = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex2:
        futs = [ex2.submit(one_s2, c) for c in todo]
        for fut in as_completed(futs):
            pending2.append(fut.result())
            n_done += 1
            if n_done % 200 == 0:
                write_merge(args.out, pending2)
                pending2 = []
                print(f"[2stage] {n_done}/{len(todo)}", flush=True)
    write_merge(args.out, pending2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
