# -*- coding: utf-8 -*-
"""schema.py — GATE-SQL 统一数据协议。

所有置信基线（B1/B2/B3/B4/B8）与评测共用同一候选池与标签格式，
键为 (qid, source, cand_idx)，与 AUC 项目的 candidates_pool.jsonl /
labels.jsonl 完全兼容（可直接复用 BIRD-300 主池）。

候选池记录（pool jsonl，每行一个 JSON）:
  {
    "qid": 10,                    # 题目 id（同一题内所有候选共享）
    "db_id": "financial",         # 数据库 id
    "difficulty": "simple",       # 可选
    "source": "nc4",              # 候选来源/采样标签（nc4/chess/safeql/reforce/macsql/
                                  #   nucleus10/beam10/qwen3-32b 等）
    "cand_idx": 0,                # 该 (qid, source) 内候选序号
    "sql": "SELECT ...",          # 候选 SQL
    "gold_sql": "SELECT ...",     # gold SQL（用于执行标注）
    "question": "...",            # NL 问题
    "evidence": "...",            # 可选 hint
    "tokens": [...],              # 可选：开源模型生成 token（B3/B4 需要）
    "token_logprobs": [...]       # 可选：与 tokens 对齐的 logprob（B3/B4 需要）
  }

标签记录（labels jsonl）:
  {"qid":10, "source":"nc4", "cand_idx":0, "label":"correct|silent|exec_fail"}
  - correct  : 严格结果集匹配 gold
  - silent   : 可执行但结果错（语义错误，判别难点）
  - exec_fail: 执行失败

置信输出（confidence jsonl，所有基线统一输出）:
  {"qid":10, "source":"nc4", "cand_idx":0, "conf":0.87, "method":"b1_verbalized"}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


@dataclass
class Candidate:
    qid: int
    source: str
    cand_idx: int
    db_id: str
    sql: str
    gold_sql: str = ""
    question: str = ""
    evidence: str = ""
    difficulty: str = ""
    tokens: Optional[list] = None
    token_logprobs: Optional[list] = None

    @property
    def key(self) -> tuple:
        return (self.qid, self.source, self.cand_idx)

    def to_row(self) -> dict:
        d = {
            "qid": self.qid, "source": self.source, "cand_idx": self.cand_idx,
            "db_id": self.db_id, "sql": self.sql, "gold_sql": self.gold_sql,
            "question": self.question, "evidence": self.evidence,
        }
        if self.difficulty:
            d["difficulty"] = self.difficulty
        if self.tokens is not None:
            d["tokens"] = self.tokens
            d["token_logprobs"] = self.token_logprobs
        return d


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_pool(path: str, require_logprobs: bool = False) -> list[Candidate]:
    """加载候选池；require_logprobs=True 时只保留带 token logprob 的记录。"""
    out: list[Candidate] = []
    for r in iter_jsonl(path):
        if require_logprobs and not r.get("token_logprobs"):
            continue
        out.append(Candidate(
            qid=int(r["qid"]), source=str(r.get("source", "pool")),
            cand_idx=int(r.get("cand_idx", 0)), db_id=str(r["db_id"]),
            sql=r["sql"], gold_sql=r.get("gold_sql", ""),
            question=r.get("question", ""), evidence=r.get("evidence", ""),
            difficulty=r.get("difficulty", ""),
            tokens=r.get("tokens"), token_logprobs=r.get("token_logprobs"),
        ))
    return out


def load_labels(path: str) -> dict[tuple, dict]:
    """返回 {(qid, source, cand_idx): {"correct": bool, "raw": str}}。

    无标签文件时返回空 dict，由调用方触发执行标注。
    """
    labels: dict[tuple, dict] = {}
    if not path or not os.path.exists(path):
        return labels
    for r in iter_jsonl(path):
        lab = str(r.get("label", "")).lower()
        labels[(int(r["qid"]), str(r["source"]), int(r["cand_idx"]))] = {
            "correct": lab == "correct", "raw": lab,
        }
    return labels


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_confidence(path: str, items: Iterable[tuple[Candidate, float]],
                     method: str) -> None:
    """统一置信输出。items: (candidate, conf) 序列。"""
    rows = ({"qid": c.qid, "source": c.source, "cand_idx": c.cand_idx,
             "conf": float(p), "method": method} for c, p in items)
    write_jsonl(path, rows)
