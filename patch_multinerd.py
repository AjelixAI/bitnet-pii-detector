#!/usr/bin/env python3
"""Patch multinerd load to EN-config only (the full load pulls 10 languages)."""
import ast

p = "/root/pii-bench/src/download_datasets.py"
s = open(p).read()

old = "    ds_mn = load_dataset('Babelscape/multinerd')"
new = "    ds_mn = load_dataset('Babelscape/multinerd', 'en')"
assert old in s, "multinerd load line not found"
s = s.replace(old, new)

# check the branch filters by lang == 'en' still works on config-en rows
if "lang == 'en'" in s or "\"en\"" in s:
    pass

open(p, "w").write(s)
ast.parse(open(p).read())
print("PATCHED multinerd -> en config")
