# -*- coding: utf-8 -*-
"""B5 LLM-judge（连续分）— 判官模型对候选打分。

对应创新点文档 §3.3 B5（LLM-as-judge）。与 B1 的区别：
  B1 问生成者"你对它的信心"；B5 问独立判官"这个 SQL 正确的概率"。
  Phase 1 用 AUC 归档二元 verdict（allow/alarm，无组内排序）；
  本版本为连续分协议（0-100），支持组内排序与 selective prediction。

输入: 候选池 jsonl + 每库 DDL 缓存目录（默认 AUC cache）
输出: confidence jsonl {"qid","source","cand_idx","conf","method"}

用法:
  python run_b5_judge.py --pool PATH --out PATH [--model gpt-4o] [--workers 10]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import load_pool, iter_jsonl, write_jsonl  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402

DEFAULT_DDL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ddl")

PROMPT = """You are a strict SQL correctness judge. Given the database schema, a natural language question and a candidate SQL query, evaluate whether the query correctly answers the question.

### Schema
{schema}

### Question
{question}
{evidence}
### Candidate SQL
{sql}

Estimate the probability (0-100) that this query returns exactly the correct answer.
Respond with ONLY a number between 0 and 100, nothing else."""


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
    if v > 1.0:  # 0-100 -> 0-1
        v /= 100.0
    return min(1.0, max(0.0, v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ddl-dir", default=DEFAULT_DDL_DIR)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    args = ap.parse_args()

    done: set[tuple] = set()
    if os.path.exists(args.out):  # 断点续跑
        for r in iter_jsonl(args.out):
            done.add((r["qid"], r["source"], r["cand_idx"]))
        print(f"[resume] 已有 {len(done)} 条")

    pool = load_pool(args.pool)
    if args.limit:
        pool = pool[: args.limit]
    todo = [c for c in pool if c.key not in done]
    print(f"[b5] 待评 {len(todo)} 条 (workers={args.workers})")

    ddl_cache: dict[str, str] = {}
    cli = LLMClient(model=args.model)

    def one(c) -> dict:
        if c.db_id not in ddl_cache:
            ddl_cache[c.db_id] = load_ddl(args.ddl_dir, c.db_id)
        ev = f"Hint: {c.evidence}\n" if c.evidence else ""
        prompt = PROMPT.format(schema=ddl_cache[c.db_id], question=c.question,
                               evidence=ev, sql=c.sql)
        try:
            text = cli.generate(prompt, n=1, temperature=0.0, max_tokens=16)[0]
            conf = parse_conf(text)
        except Exception as e:  # 单条失败不中断
            print(f"[warn] qid={c.qid} {e}", flush=True)
            conf = 0.5
        return {"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                "conf": conf, "method": "b5_judge_cont"}

    pending: list[dict] = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for fut in as_completed(futs):
            pending.append(fut.result())
            n_done += 1
            if n_done % 200 == 0:
                write_jsonl(args.out, _merge(args.out, pending))
                pending = []
                print(f"[b5] {n_done}/{len(todo)}", flush=True)
    if n_done:
        write_jsonl(args.out, _merge(args.out, pending))
    print(f"[done] -> {args.out}")


def _merge(out_path: str, rows: list) -> list:
    old = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            old = [json.loads(l) for l in f if l.strip()]
    seen = {(r["qid"], r["source"], r["cand_idx"]) for r in old}
    return old + [r for r in rows
                  if (r["qid"], r["source"], r["cand_idx"]) not in seen]


if __name__ == "__main__":
    main()
