# -*- coding: utf-8 -*-
"""run_b1_cross.py — 通用 verbalized-confidence（支持异构端点/模型 + 温度覆盖）。

与 b1_verbalized/run_b1.py 同协议（0-100 自报置信），但支持：
  - --base-url / --api-key-env 显式端点覆盖（如 Moonshot kimi）
  - --temperature / --top-p 覆盖（Kimi 非思考模式强制 T=0.6, top_p=0.95）
  - --reasoning-effort none（Moonshot Kimi 关闭思考）
  - --pool 传入任意候选池（如第二生成器的 eval 子集池）

用法:
  python run_b1_cross.py --pool PATH --out PATH --model kimi-k2.6 \
      --base-url https://api.moonshot.cn/v1 --api-key-env MOONSHOT_API_KEY \
      --temperature 0.6 --top-p 0.95 --reasoning-effort none
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
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b1_verbalized"))
from common.schema import load_pool, iter_jsonl  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402
from run_b1 import PROMPT, load_ddl, parse_conf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ddl-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ddl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="b1_verbalized")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    kw = {}
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.api_key_env:
        kw["api_key"] = os.environ[args.api_key_env]
    cli = LLMClient(model=args.model, **kw)
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None

    pool = load_pool(args.pool)
    done = set()
    if os.path.exists(args.out):
        done = {(r["qid"], r["source"], r["cand_idx"])
                for r in iter_jsonl(args.out)}
        print(f"[resume] 已有 {len(done)} 条")
    todo = [c for c in pool if (c.qid, c.source, c.cand_idx) not in done]
    print(f"[b1cross] model={args.model} T={args.temperature} top_p={args.top_p} "
          f"待评 {len(todo)} 条")

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
                                top_p=args.top_p, max_tokens=16,
                                extra=extra)[0]
            conf = parse_conf(text)
            if not re.search(r"\d", text):
                n_parse_fail += 1
        except Exception as e:
            print(f"[warn] qid={c.qid} {e}", flush=True)
            conf = 0.5
        return {"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
                "conf": conf, "method": args.method}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def flush(rows):
        old = []
        if os.path.exists(args.out):
            old = [json.loads(l) for l in open(args.out, encoding="utf-8")
                   if l.strip()]
        seen = {(r["qid"], r["source"], r["cand_idx"]) for r in old}
        merged = old + [r for r in rows
                        if (r["qid"], r["source"], r["cand_idx"]) not in seen]
        with open(args.out, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pending = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for fut in as_completed(futs):
            pending.append(fut.result())
            n_done += 1
            if n_done % 200 == 0:
                flush(pending)
                pending = []
                print(f"[b1cross] {n_done}/{len(todo)}", flush=True)
    flush(pending)
    print(f"[done] parse_fail={n_parse_fail} -> {args.out}")


if __name__ == "__main__":
    main()
