#!/usr/bin/env python3
"""Patch pii-bench download script: local parquet loaders for wikiann/conll/finer.
Idempotent. Also ensures finer_test.parquet exists."""
import os
import shutil

from huggingface_hub import hf_hub_download

OUT = "/root/pii-bench/pii_datasets/_raw_parquet"
os.makedirs(OUT, exist_ok=True)


def ensure(fname, repo, remote, rev=None):
    dst = os.path.join(OUT, fname)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1024:
        p = hf_hub_download(repo, remote, repo_type="dataset", revision=rev)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        shutil.copy(p, dst)
        print(f"downloaded {fname}")
    else:
        print(f"have {fname} ({os.path.getsize(dst)//1024} KB)")


ensure("finer_test.parquet", "nlpaueb/finer-139", "finer-139/test/0000.parquet",
       rev="refs/convert/parquet")

WL = "/root/pii-bench/src/download_datasets.py"
lines = open(WL).read().split("\n")

REPL = {
    249: "    ds_wikiann = DatasetDict({'train': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/wikiann_en_train.parquet'), 'validation': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/wikiann_en_validation.parquet'), 'test': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/wikiann_en_test.parquet')})",
    366: "    ds_conll = DatasetDict({'train': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/conll_train.parquet'), 'validation': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/conll_validation.parquet'), 'test': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/conll_test.parquet')})",
    398: "    ds_finer = DatasetDict({'train': Dataset.from_parquet(['/root/pii-bench/pii_datasets/_raw_parquet/finer_train0.parquet', '/root/pii-bench/pii_datasets/_raw_parquet/finer_train1.parquet']), 'validation': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/finer_val.parquet'), 'test': Dataset.from_parquet('/root/pii-bench/pii_datasets/_raw_parquet/finer_test.parquet')})",
}
for ln, new in REPL.items():
    old = lines[ln - 1]
    lines[ln - 1] = new
    assert "load_dataset" in old, f"line {ln} not a load_dataset: {old[:80]}"
    print(f"line {ln} patched: {old[:60]}...")

for i, l in enumerate(lines):
    if l.startswith("from datasets import load_dataset"):
        lines[i] = "from datasets import load_dataset, DatasetDict, Dataset"
        print("import patched")
        break

open(WL, "w").write("\n".join(lines))
import ast
ast.parse(open(WL).read())
print("SYNTAX OK")
