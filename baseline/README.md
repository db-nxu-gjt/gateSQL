# Confidence baselines (B0-B8)

Every confidence method audited in the paper, under one shared data protocol.

```
baseline/
├── common/
│   ├── schema.py            data protocol (pool / labels / conf jsonl)
│   ├── exec_match.py        strict result-set matching (sorted-tuple equality)
│   ├── metrics.py           AUROC / ECE / Brier / NLL / AURC / SelEX / cluster bootstrap
│   └── llm_client.py        OpenAI-compatible client (n-sampling + token logprobs;
│                            extra payload passthrough, e.g. reasoning_effort=none)
├── b1_verbalized/run_b1.py       B1 verbalized confidence (0-100, greedy)
├── b2_consistency/run_b2.py      B2 execution-consistency vote (k samples)
├── b3_sac_ftc/run_b3_sac.py      B3a SAC-Avg (execution-aware token scoring)
├── b3_sac_ftc/run_b3_ftc.py      B3b FTC-Product
├── b4_recalib/run_b4_recalib.py  B4 logistic recalibration (clause features + z)
├── b5_judge/run_b5_judge.py      B5 continuous LLM-judge (0-100 probability)
├── b6_laya/run_b6_judge.py       B6/B7 7-question questionnaire judge
│                                 (internal system, evaluated zero-shot)
├── b8_conformal/run_b8_conformal.py  B8 split-conformal auto-execution gate
├── b8_conformal/ks_test.py           KS exchangeability test
└── eval/run_eval.py              metrics over (pool, labels, conf) triples
```

Data protocol (`common/schema.py`): candidate `{qid, source, cand_idx, db_id,
sql, gold_sql, question, evidence, [tokens, token_logprobs]}`; label
`{qid, source, cand_idx, label: correct|silent|exec_fail}`; confidence
`{qid, source, cand_idx, conf, method}`. All released files under
`data/pools/` follow it.

Environment: `GATE_LLM_BASE_URL` / `GATE_LLM_API_KEY` / `GATE_LLM_MODEL`
(OpenAI-compatible endpoint), `GATE_BIRD_DBS` (default `data/bird/databases`).
Moonshot Kimi endpoints require non-thinking mode with enforced decoding
(T=0.6, top_p=0.95, n=1, `reasoning_effort=none`); the cross-endpoint runner
scripts expose `--temperature/--top-p/--reasoning-effort` for this.

All released confidence files are in `data/pools/`; `baseline/tools/p*_*.py`
regenerate every table of the paper from them without any API call (see the
top-level README).
