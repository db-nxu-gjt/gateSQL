# -*- coding: utf-8 -*-
"""dump_xlsx.py — 将 paper/results/*.xlsx 全部 sheet 打印为 TSV（核对用）。"""
import glob
import os
import sys

from openpyxl import load_workbook

OUT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results"))
for fp in sorted(glob.glob(os.path.join(OUT, "*.xlsx"))):
    print("=====", os.path.basename(fp))
    wb = load_workbook(fp, read_only=True)
    for ws in wb.worksheets:
        print("--- sheet:", ws.title)
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else
                     ("%.4f" % v if isinstance(v, float) else str(v))
                     for v in row]
            if any(c.strip() for c in cells):
                print("\t".join(cells))
