# -*- coding: utf-8 -*-
"""run_b5_cross.py — 通用 LLM-judge（支持异构端点/模型 + 温度覆盖）。

与 b5_judge/run_b5_judge.py 同协议（连续 0-100 打分），但：
  - --base-url / --api-key 显式覆盖端点（如 Moonshot kimi）
  - --temperature 可设（Moonshot Kimi 系仅允许 T=1，报告时披露）
  - --qids-json 只评指定题目（seed-0 eval 子集用）

用法:
  python run_b5_cross.py --pool PATH --out PATH --model kimi-k3 \
      --base-url https://api.moonshot.cn/v1 --api-key-env MOONSHOT_API_KEY \
      --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
from common.schema import load_pool, iter_jsonl  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402

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
    if v > 1.0:
        v /= 100.0
    return min(1.0, max(0.0, v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ddl-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ddl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="b5_cross")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None,
                    help="API key 所在环境变量名")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", default=None,
                    help="如 none（Moonshot Kimi 关闭思考）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--qids-json", default=None)
    args = ap.parse_args()

    kw = {}
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.api_key_env:
        kw["api_key"] = os.environ[args.api_key_env]
    cli = LLMClient(model=args.model, **kw)
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None

    keep = None
    if args.qids_json:
        keep = set(json.load(open(args.qids_json, encoding="utf-8")))

    pool = load_pool(args.pool)
    if keep is not None:
        pool = [c for c in pool if c.qid in keep]

    done = set()
    if os.path.exists(args.out):
        done = {(r["qid"], r["source"], r["cand_idx"])
                for r in iter_jsonl(args.out)}
        print(f"[resume] 已有 {len(done)} 条")
    todo = [c for c in pool if (c.qid, c.source, c.cand_idx) not in done]
    print(f"[b5cross] model={args.model} T={args.temperature} "
          f"待评 {len(todo)} 条 (workers={args.workers})")

    ddl_cache = {}
    n_parse_fail = 0

    def one(c) -> dict:
        nonlocal n_parse_fail
        if c.db_id not in ddl_cache:
            ddl_cache[c.db_id] = load_ddl(args.ddl_dir, c.db_id)
        ev = f"Hint: {c.evidence}\n" if c.evidence else ""
        prompt = PROMPT.format(schema=ddl_cache[c.db_id], question=c.question,
                               evidence=ev, sql=c.sql)
        try:
            text = cli.generate(prompt, n=1, temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=16, extra=extra)[0]
            conf = parse_conf(text)
            if not re.search(r"\d", text):
                n_parse_fail += 1
        except Exception as e:
            print(f"[warn] qid={c.qid} {e}", flush=True)
            conf = 0.5
        return {"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                "conf": conf, "method": args.method}

    pending = []
    n_done = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def flush():
        old = []
        if os.path.exists(args.out):
            old = [json.loads(l) for l in open(args.out, encoding="utf-8")
                   if l.strip()]
        seen = {(r["qid"], r["source"], r["cand_idx"]) for r in old}
        merged = old + [r for r in pending
                        if (r["qid"], r["source"], r["cand_idx"]) not in seen]
        with open(args.out, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for fut in as_completed(futs):
            pending.append(fut.result())
            n_done += 1
            if n_done % 200 == 0:
                flush()
                pending = []
                print(f"[b5cross] {n_done}/{len(todo)}", flush=True)
    flush()
    print(f"[done] parse_fail={n_parse_fail} -> {args.out}")


if __name__ == "__main__":
    main()
