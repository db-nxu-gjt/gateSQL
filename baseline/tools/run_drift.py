# -*- coding: utf-8 -*-
"""run_drift.py — C2: schema 漂移实验（rename / add_col / split_tbl）。

协议：
  层1（本地）: 为有效变体构建漂移 sqlite（数据迁移 + VIEW 兼容），
    候选/gold 在漂移库重标注（rename 按 meta.rename_map 词法改写 SQL，
    并以"标签与原库一致率"作改写正确性 sanity）；
    B4/fuse 用原库全局模型对漂移 SQL 特征重打分（logprob/share 保持原值，
    受控测度：分数仅随词法/结构特征变化）。
  层2（API）: B1 verbalized 判官在漂移 DDL prompt 下重评涉及候选
    （rename=改写 SQL 见新 DDL；split=原 SQL 见碎片化 schema+VIEW），
    测判官对 DDL 漂移的适应/困惑。

用法:
  python run_drift.py --layer build    # 构建 24 个漂移库
  python run_drift.py --layer exec     # 重标注 + B4/fuse 重打分 + 汇总
  python run_drift.py --layer b1       # B1 判官漂移重评（API）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, os.path.join(_ROOT, "baseline", "b4_recalib"))
from common.schema import load_pool, load_labels  # noqa: E402
from common import metrics as M  # noqa: E402
from common.exec_match import run_sql  # noqa: E402
from common.llm_client import LLMClient  # noqa: E402
from run_b4_recalib import clause_features, fit_logistic  # noqa: E402

DRIFT = os.path.normpath(os.path.join(_ROOT, "data", "schema_drift"))
BIRD_DBS = os.environ.get("GATE_BIRD_DBS") or os.path.join(
    _ROOT, "data", "bird", "databases")
BIRD_DBS = os.path.normpath(BIRD_DBS)
CACHE = os.path.join(_ROOT, "data", "ddl")
POOL = os.path.join(_ROOT, "data", "pools", "bird300_ds_pool.jsonl")
LABELS = os.path.join(_ROOT, "data", "pools", "bird300_ds_labels.jsonl")
SHARES = [os.path.join(_ROOT, "data", "pools", f"bird300_ds_b2_{s}.jsonl")
          for s in ("nuc10", "beam10")]
B1_CONF = os.path.join(_ROOT, "data", "pools", "bird300_ds_b1.jsonl")

B1_PROMPT = """You are given a database schema, a natural language question and a candidate SQL query.

### Schema
{schema}

### Question
{question}
{evidence}
### Candidate SQL
{sql}

How confident are you that this SQL query correctly answers the question?
Respond with ONLY a number between 0 and 100 (your confidence), nothing else."""


def valid_variants() -> list[tuple[str, str]]:
    out = []
    for db in sorted(os.listdir(DRIFT)):
        for p in ("rename", "add_col", "split_tbl"):
            fp = os.path.join(DRIFT, db, p, "meta.json")
            if not os.path.exists(fp):
                continue
            m = json.load(open(fp, encoding="utf-8"))
            meta = m.get("meta") or {}
            if p == "split_tbl" and not meta.get("split"):
                continue
            if p == "rename" and not meta.get("rename_map"):
                continue
            out.append((db, p))
    return out


def _norm_ddl(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def build_one(db: str, pert: str) -> str:
    """构建漂移 sqlite，返回路径。数据迁移：rename 按 map，add_col 新列 NULL，
    split 的 main/ext 表按列名交集复制。"""
    meta = json.load(open(os.path.join(DRIFT, db, pert, "meta.json"),
                          encoding="utf-8"))["meta"]
    schema = json.load(open(os.path.join(DRIFT, db, pert, "schema.json"),
                            encoding="utf-8"))
    orig_ddl = json.load(open(os.path.join(CACHE, f"ddl_{db}.json"),
                              encoding="utf-8"))
    out_path = os.path.join(DRIFT, db, pert, "drift.sqlite")
    if os.path.exists(out_path):
        return out_path

    orig = sqlite3.connect(os.path.join(BIRD_DBS, db, db + ".sqlite"))
    con = sqlite3.connect(out_path)
    orig.backup(con)
    con.commit()

    rename_map = meta.get("rename_map") or {}
    new2old = {}
    for k, v in rename_map.items():
        ot, oc = k.split(".", 1)
        new2old[(ot, v)] = oc
    split = meta.get("split") or {}
    rebuild = set()
    for t, ddl in schema.items():
        if t.lower().startswith("sqlite_"):
            continue
        if _norm_ddl(orig_ddl.get(t, "")) != _norm_ddl(ddl):
            rebuild.add(t)
    # split: 目标表被 drop（由 main/ext + VIEW 替代）
    if split:
        rebuild.add(split["table"])

    snap = {}
    for t in rebuild:
        try:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            rows = con.execute(f'SELECT * FROM "{t}"').fetchall()
            idx_sql = [r[0] for r in con.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND sql IS NOT NULL AND tbl_name = ?", (t,))]
            snap[t] = (cols, rows, idx_sql)
        except sqlite3.OperationalError:
            snap[t] = ([], [], [])
    for t in rebuild:
        for stmt in (f'DROP VIEW IF EXISTS "{t}"',
                     f'DROP TABLE IF EXISTS "{t}"'):
            try:
                con.execute(stmt)
            except sqlite3.OperationalError:
                pass  # view/table 类型不符时由另一条语句处理
    views = []
    for t, ddl in schema.items():
        if t not in rebuild:
            continue
        if ddl.lstrip().lower().startswith("create view"):
            views.append(t)
            continue
        con.execute(ddl)
    for t in views:
        con.execute(schema[t])

    for t in rebuild:
        if t in views or not snap.get(t) or not snap[t][0]:
            continue
        old_cols, rows, idx_sql = snap[t]
        new_cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        col_map = []
        for nc in new_cols:
            if (t, nc) in new2old:
                col_map.append(new2old[(t, nc)])
            elif nc in old_cols:
                col_map.append(nc)
            else:
                col_map.append(None)  # add_col 新列 → NULL
        if not rows:
            continue
        qm = ",".join("?" * len(new_cols))
        # 用列名占位将 NULL 并入 SELECT：直接逐行构造
        data = []
        idx = {c: i for i, c in enumerate(old_cols)}
        for r in rows:
            data.append(tuple(None if c is None else r[idx[c]] for c in col_map))
        con.executemany(
            f'INSERT INTO "{t}" ({",".join(chr(34) + c + chr(34) for c in new_cols)}) '
            f"VALUES ({qm})", data)
        # 重建表恢复原索引（rename 表的索引列同步改写，避免全表扫）
        old2new = {oc: nc for (tt, oc), nc in new2old.items() if tt == t}
        for isql in idx_sql:
            s = isql
            for oc, nc in old2new.items():
                s = re.sub(rf"\b{re.escape(oc)}\b", nc, s, flags=re.I)
            try:
                con.execute(s)
            except sqlite3.OperationalError:
                pass
    con.commit()
    con.close()
    orig.close()
    return out_path


def rewrite_map(db: str, pert: str):
    """rename 精确映射。返回 (precise, col2tables)：
    precise: {(table, old_col): new_col}；col2tables: {old_col: set(tables)}
    （原库列 -> 表集合，用 cache DDL 解析，供无限定列的归属判定）。"""
    meta = json.load(open(os.path.join(DRIFT, db, pert, "meta.json"),
                          encoding="utf-8"))["meta"]
    precise = {}
    for k, v in (meta.get("rename_map") or {}).items():
        t, c = k.split(".", 1)
        precise[(t.lower(), c.lower())] = v
    col2tables: dict = {}
    con = sqlite3.connect(os.path.join(BIRD_DBS, db, db + ".sqlite"))
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            for r in con.execute(f'PRAGMA table_info("{t}")'):
                col2tables.setdefault(r[1].lower(), set()).add(t.lower())
    finally:
        con.close()
    return precise, col2tables


def rewrite_sql(sql: str, precise: dict, col2tables: dict) -> str:
    """表感知列名改写：解析 SQL，对每个列引用判定所属表后仅替换确被改名的
    列（避免误伤其他表同名列）。解析失败退回全局词法替换。"""
    import sqlglot
    from sqlglot import exp
    old_cols = {c for (_, c) in precise}
    if not precise:
        return sql
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        out = sql
        for (_, oc), nc in precise.items():
            out = re.sub(rf"\b{re.escape(oc)}\b", nc, out, flags=re.I)
        return out
    alias2tbl = {}
    sql_tables = set()
    for t in tree.find_all(exp.Table):
        if not t.name:
            continue
        sql_tables.add(t.name.lower())
        if (t.alias or "").lower():
            alias2tbl[t.alias.lower()] = t.name.lower()
    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if name not in old_cols:
            continue
        tbl = (col.table or "").lower()
        if tbl:
            ctx = {alias2tbl.get(tbl, tbl)}
        else:  # 无表限定：列在 SQL 表集合中的归属
            ctx = col2tables.get(name, set()) & sql_tables or \
                col2tables.get(name, set())
        news = {precise[(t, name)] for t in ctx if (t, name) in precise}
        if len(news) == 1:
            col.set("this", exp.to_identifier(news.pop()))
    return tree.sql(dialect="sqlite")


def _global_b4_model():
    """原库全局 B4 + fuse L2 模型（run_b4_recalib / run_fuse 同协议 seed 0）。"""
    pool = load_pool(POOL, require_logprobs=True)
    labels = load_labels(LABELS)
    data = []
    for c in pool:
        lab = labels.get(c.key)
        if lab is None:
            continue
        z = sum(c.token_logprobs) if c.token_logprobs else 0.0
        data.append((c, z, clause_features(c.sql), lab["correct"]))
    shares: dict = {}
    for f in SHARES:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    shares[(r["qid"], r["source"], r["cand_idx"])] = r["conf"]
    qids = sorted({c.qid for c, _, _, _ in data})
    rng = random.Random(0)
    rng.shuffle(qids)
    n_fit = max(1, int(len(qids) * 0.3))
    fit_qids = set(qids[:n_fit])
    fit_idx = [i for i, d in enumerate(data) if d[0].qid in fit_qids]
    dim = len(data[0][2])
    mu = [sum(data[i][2][j] for i in fit_idx) / len(fit_idx) for j in range(dim)]
    sd = [max(1e-6, math.sqrt(sum((data[i][2][j] - mu[j]) ** 2 for i in fit_idx)
                            / len(fit_idx))) for j in range(dim)]

    def make_x(i):
        return ([(fj - m) / s for fj, m, s in zip(data[i][2], mu, sd)]
                + [data[i][1] / 10.0])

    params_l1 = fit_logistic([make_x(i) for i in fit_idx],
                             [data[i][3] for i in fit_idx])

    def raw_of(feat, z):
        x = [(fj - m) / s for fj, m, s in zip(feat, mu, sd)] + [z / 10.0]
        zz = params_l1[-1] + sum(w * xj for w, xj in zip(params_l1[:-1], x))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

    # L2 OOF
    def logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    fq = sorted(fit_qids)
    rng2 = random.Random(1)
    rng2.shuffle(fq)
    half = max(1, len(fq) // 2)
    folds = (set(fq[:half]), set(fq[half:]))
    oof = {}
    for ftr, fte in ((folds[0], folds[1]), (folds[1], folds[0])):
        tri = [i for i in fit_idx if data[i][0].qid in ftr]
        tei = [i for i in fit_idx if data[i][0].qid in fte]
        p_ = fit_logistic([make_x(i) for i in tri],
                          [data[i][3] for i in tri])
        for i in tei:
            oof[i] = raw_of(data[i][2], data[i][1])

    def l2_x(s, sh):
        return [logit(s), sh, logit(s) * sh]

    l2_fit = [i for i in fit_idx if i in oof]
    l2_raw = [l2_x(oof[i], shares.get(data[i][0].key, 0.5)) for i in l2_fit]
    l2_y = [1.0 if data[i][3] else 0.0 for i in l2_fit]
    d2 = len(l2_raw[0])
    mu2 = [sum(r[j] for r in l2_raw) / len(l2_raw) for j in range(d2)]
    sd2 = [max(1e-6, math.sqrt(sum((r[j] - mu2[j]) ** 2 for r in l2_raw)
                            / len(l2_raw))) for j in range(d2)]
    params_l2 = fit_logistic(
        [[(r[j] - mu2[j]) / sd2[j] for j in range(d2)] for r in l2_raw], l2_y)

    def fuse_of(feat, z, sh):
        x = l2_x(raw_of(feat, z), sh)
        zz = params_l2[-1] + sum(w * xj for w, xj in
                                 zip(params_l2[:-1],
                                     [(xj - m) / s
                                      for xj, m, s in zip(x, mu2, sd2)]))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, zz))))

    return data, raw_of, fuse_of


def layer_exec() -> None:
    data, raw_of, fuse_of = _global_b4_model()
    orig_lab = {}
    with open(LABELS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                orig_lab[(r["qid"], r["source"], r["cand_idx"])] = r["label"]
    shares: dict = {}
    for f in SHARES:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    shares[(r["qid"], r["source"], r["cand_idx"])] = r["conf"]

    print("| variant | n | 标签一致率 | B4-drift AUROC | B4 AUROC(原) "
          "| fuse-drift AUROC | fuse AUROC(原) | B4-drift ECE |")
    print("|---|---|---|---|---|---|---|---|")
    for db, pert in valid_variants():
        dpath = build_one(db, pert)
        precise, col2tables = rewrite_map(db, pert) if pert == "rename" \
            else ({}, {})
        sub = [d for d in data if d[0].db_id == db]
        agree = n = 0
        sc_b4, sc_fuse, co = [], [], []
        for c, z, feat, corr in sub:
            sql_d = rewrite_sql(c.sql, precise, col2tables) if precise else c.sql
            gold_d = rewrite_sql(c.gold_sql, precise, col2tables) \
                if precise else c.gold_sql
            pred = run_sql(dpath, sql_d)
            if pred is None:
                lab_d = "exec_fail"
            else:
                gold = run_sql(dpath, gold_d)
                lab_d = "exec_fail" if gold is None else (
                    "correct" if pred == gold else "silent")
            n += 1
            if lab_d == orig_lab.get(c.key):
                agree += 1
            f_d = clause_features(sql_d)
            sc_b4.append(raw_of(f_d, z))
            sc_fuse.append(fuse_of(f_d, z, shares.get(c.key, 0.5)))
            co.append(lab_d == "correct")
        # 受控对照：同一批候选用**原 SQL** 特征的原库分数
        co_orig = [corr for _, _, _, corr in sub]
        au_b4_orig = M.auroc([raw_of(f, z) for _, z, f, _ in sub], co_orig)
        au_fuse_orig = M.auroc(
            [fuse_of(f, z, shares.get(c.key, 0.5))
             for c, z, f, _ in sub], co_orig)
        au_b4, au_fuse = M.auroc(sc_b4, co), M.auroc(sc_fuse, co)
        ece_b4 = M.ece(sc_b4, co)
        print("| %s/%s | %d | %.4f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            db, pert, n, agree / n, au_b4, au_b4_orig, au_fuse,
            au_fuse_orig, ece_b4))


def layer_b1() -> None:
    pool = load_pool(POOL)
    orig_b1 = {}
    with open(B1_CONF, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                orig_b1[(r["qid"], r["source"], r["cand_idx"])] = float(r["conf"])
    labels = load_labels(LABELS)
    cli = LLMClient()
    out_rows = []
    for db, pert in valid_variants():
        if pert not in ("rename", "split_tbl"):
            continue  # add_col 不改判官可见结构（新列 DEFAULT NULL 且 SQL 不引用）
        meta = json.load(open(os.path.join(DRIFT, db, pert, "meta.json"),
                              encoding="utf-8"))["meta"]
        schema_txt = "\n\n".join(
            json.load(open(os.path.join(DRIFT, db, pert, "schema.json"),
                           encoding="utf-8")).values())
        cmap = rewrite_map(db, pert) if pert == "rename" else ({}, {})
        # rename 用**旧列名**筛原 SQL 中涉及的候选；split 用被拆表名
        targets = {oc for (_, oc) in cmap[0]} if cmap[0] \
            else {meta["split"]["table"]}
        sub = [c for c in pool if c.db_id == db
               and any(re.search(rf"\b{re.escape(t)}\b", c.sql, re.I)
                       for t in targets)]
        sub = sub[:60]
        print(f"[b1-drift] {db}/{pert}: {len(sub)} 候选")

        def work(c):
            sql_d = rewrite_sql(c.sql, *cmap) if cmap[0] else c.sql
            ev = f"Hint: {c.evidence}\n" if c.evidence else ""
            p = B1_PROMPT.format(schema=schema_txt, question=c.question,
                                 evidence=ev, sql=sql_d)
            for _ in range(3):
                try:
                    txt = cli.generate(p, n=1, temperature=0.0,
                                       max_tokens=16)[0].strip()
                    m = re.search(r"(\d+(?:\.\d+)?)", txt)
                    if m:
                        v = float(m.group(1))
                        return c, (v / 100.0 if v > 1.0 else v)
                except Exception:
                    pass
                import time
                time.sleep(1)
            return c, None

        with ThreadPoolExecutor(max_workers=8) as ex:
            for c, conf in ex.map(work, sub):
                if conf is None:
                    continue
                out_rows.append({"qid": c.qid, "source": c.source,
                                 "cand_idx": c.cand_idx, "db_id": db,
                                 "pert": pert, "conf": round(conf, 4),
                                 "conf_orig": orig_b1.get(c.key),
                                 "label_orig": labels.get(c.key, {}).get("label")})
    out = os.path.join(_ROOT, "data", "pools", "bird300_ds_b1_drift.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(out_rows)} -> {out}")

    for pert in ("rename", "split_tbl"):
        rows = [r for r in out_rows if r["pert"] == pert
                and r["conf_orig"] is not None
                and r["label_orig"] in ("correct", "silent")]
        if not rows:
            continue
        co = [r["label_orig"] == "correct" for r in rows]
        au_d = M.auroc([r["conf"] for r in rows], co)
        au_o = M.auroc([r["conf_orig"] for r in rows], co)
        mad = sum(abs(r["conf"] - r["conf_orig"]) for r in rows) / len(rows)
        print(f"# B1-drift {pert}: n={len(rows)} AUROC {au_o:.3f}->{au_d:.3f} "
              f"| ECE {M.ece([r['conf_orig'] for r in rows], co):.3f}"
              f"->{M.ece([r['conf'] for r in rows], co):.3f} "
              f"| mean|Δconf|={mad:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=["build", "exec", "b1"])
    args = ap.parse_args()
    if args.layer == "build":
        for db, pert in valid_variants():
            p = build_one(db, pert)
            print(f"[build] {db}/{pert} -> {p}")
    elif args.layer == "exec":
        layer_exec()
    else:
        layer_b1()


if __name__ == "__main__":
    main()
