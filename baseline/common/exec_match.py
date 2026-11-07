# -*- coding: utf-8 -*-
"""exec_match.py — 严格结果集匹配（口径移植自 RGCV / AUC sqlres.py）。

口径：sorted(tuple(rows)) 严格相等；异常/超时 -> None（exec_fail）。
数据库根目录默认 <repo>/data/bird/databases（BIRD dev 数据库需按 data/bird/README.md
自行下载后解压至此），可用环境变量覆盖：
    GATE_BIRD_DBS   （默认 <repo>/data/bird/databases）
    GATE_SPIDER_DBS （可选；默认 <repo>/data/spider/databases）
"""
from __future__ import annotations

import os
import sqlite3
import threading
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIRD_DBS = os.environ.get("GATE_BIRD_DBS", os.path.join(_ROOT, "data", "bird", "databases"))
SPIDER_DBS = os.environ.get("GATE_SPIDER_DBS", os.path.join(_ROOT, "data", "spider", "databases"))


def db_path_of(db_id: str, dataset: str = "bird") -> str:
    root = BIRD_DBS if dataset == "bird" else SPIDER_DBS
    return os.path.join(root, db_id, db_id + ".sqlite")


def run_sql(db_path: str, sql: str, timeout: int = 10, watchdog: float = 30.0):
    """执行 SQL，返回 sorted(tuple) 结果集；失败/超时返回 None（口径同 RGCV）。"""
    conn = None
    timer = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        if watchdog:
            timer = threading.Timer(watchdog, conn.interrupt)
            timer.daemon = True
            timer.start()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return sorted(tuple(r) for r in rows)
    except Exception:
        return None
    finally:
        if timer:
            timer.cancel()
            try:
                conn.close()  # type: ignore[union-attr]
            except Exception:
                pass


def label_candidate(cand, dataset: str = "bird") -> str:
    """对单个候选执行标注：correct / silent / exec_fail。"""
    db = db_path_of(cand.db_id, dataset)
    if not os.path.exists(db):
        return "exec_fail"
    pred = run_sql(db, cand.sql)
    if pred is None:
        return "exec_fail"
    gold = run_sql(db, cand.gold_sql)
    if gold is None:
        return "exec_fail"
    return "correct" if pred == gold else "silent"


def label_pool(cands, dataset: str = "bird", cache_path: Optional[str] = None,
               log_every: int = 200) -> dict[tuple, str]:
    """批量标注（带断点续跑缓存）。返回 {(qid, source, cand_idx): label}。"""
    cache: dict[tuple, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = __import__("json").loads(line)
                    cache[(r["qid"], r["source"], r["cand_idx"])] = r["label"]
    rows = []
    out = dict(cache)
    fout = None
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        fout = open(cache_path, "a", encoding="utf-8")
    try:
        for i, c in enumerate(cands):
            if c.key in out:
                continue
            lab = label_candidate(c, dataset)
            out[c.key] = lab
            rows.append({"qid": c.qid, "source": c.source,
                         "cand_idx": c.cand_idx, "label": lab})
            if fout:
                fout.write(__import__("json").dumps(rows[-1]) + "\n")
                fout.flush()
            if (i + 1) % log_every == 0:
                print(f"[exec_label] {i + 1}/{len(cands)}")
    finally:
        if fout:
            fout.close()
    return out
