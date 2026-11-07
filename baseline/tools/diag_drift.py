# -*- coding: utf-8 -*-
"""诊断 california_schools/rename 改写后执行失败的原因。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "baseline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sqlite3  # noqa: E402
from run_drift import build_one, rewrite_map, rewrite_sql, POOL  # noqa: E402
from common.schema import load_pool  # noqa: E402

db, pert = "california_schools", "rename"
dpath = build_one(db, pert)
precise, col2tables = rewrite_map(db, pert)
print("precise:", precise)
print("drift cols:")
con = sqlite3.connect(dpath)
for t, in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
    print(" ", t, cols)

pool = [c for c in load_pool(POOL) if c.db_id == db][:200]
n_shown = 0
for c in pool:
    sql_d = rewrite_sql(c.sql, precise, col2tables)
    gold_d = rewrite_sql(c.gold_sql, precise, col2tables)
    try:
        con.execute(sql_d).fetchall()
        err_c = None
    except Exception as e:
        err_c = str(e)
    try:
        con.execute(gold_d).fetchall()
        err_g = None
    except Exception as e:
        err_g = str(e)
    if (err_c or err_g) and n_shown < 4:
        n_shown += 1
        print("\n--- orig:", c.sql[:160])
        print("--- rew :", sql_d[:160])
        print("--- err_c:", err_c)
        print("--- err_g:", err_g)
print("\nrewritten sample:", rewrite_sql(pool[0].sql, precise, col2tables))
