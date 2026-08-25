#!/usr/bin/env python3
# build_pretrain_corpus_mp.py — MULTI-PROCESS FineWeb -> token-ID shard builder.
#
# Producer streams FineWeb docs (from cache), N worker processes tokenize in
# parallel (24 cores), a writer appends packed docs to shard files. This is ~8-12x
# faster than single-threaded and lets us reach ~2B tokens in ~1-1.5h.
#
# Output: base.bin, base_shard1.bin, ... each uint32 [len, ids...] per doc.
import argparse, os, sys, time
import numpy as np
from multiprocessing import Process, Queue, Value

TOK_PATH = "/root/pii_data/pretok_gpt/tokenizer.json"
SEP = None

def load_tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOK_PATH)

def worker(work_q, out_q, seq_cap, worker_id):
    tok = load_tok()
    while True:
        item = work_q.get()
        if item is None:
            break
        idx, text = item
        if not text or len(text.strip()) < 30:
            out_q.put((idx, None))
            continue
        e = tok.encode(text)
        ids = e.ids[:seq_cap]
        out_q.put((idx, ids if len(ids) >= 3 else None))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/pii_corpus/pretok_corpus.bin")
    ap.add_argument("--max-docs", type=int, default=20_000_000)
    ap.add_argument("--seq-cap", type=int, default=512)
    ap.add_argument("--min-docs-per-shard", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    workers = args.workers or (os.cpu_count() or 8)
    print(f"building with {workers} workers", flush=True)

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)

    work_q = Queue(maxsize=2000)
    out_q = Queue(maxsize=4000)
    procs = [Process(target=worker, args=(work_q, out_q, args.seq_cap, w)) for w in range(workers)]
    for p in procs: p.start()

    total_written = 0
    shard_idx = 0
    cur_shard_docs = []
    t0 = time.time()

    def flush():
        nonlocal cur_shard_docs, shard_idx
        if not cur_shard_docs:
            return
        p = args.out if shard_idx == 0 else args.out.replace(".bin", f"_shard{shard_idx}.bin")
        arrs = [np.array([len(ids)] + ids, dtype=np.uint32) for ids in cur_shard_docs]
        with open(p, "wb") as f:
            f.write(np.concatenate(arrs).tobytes())
        print(f"wrote {p} ({len(cur_shard_docs):,} docs)", flush=True)
        cur_shard_docs = []
        shard_idx += 1

    it = iter(ds)
    doc_idx = 0
    in_flight = 0
    try:
        while total_written < args.max_docs:
            # feed
            while in_flight < workers * 8 and total_written + in_flight < args.max_docs:
                try:
                    d = next(it)
                except StopIteration:
                    break
                text = d.get("text", "")
                work_q.put((doc_idx, text))
                doc_idx += 1
                in_flight += 1
            # collect
            got = 0
            while got < in_flight:
                idx, ids = out_q.get()
                in_flight -= 1
                got += 1
                if ids is not None:

                    total_written += 1
                    cur_shard_docs.append(ids)
                    if len(cur_shard_docs) >= args.min_docs_per_shard:
                        flush()
                    if total_written % 50000 == 0:
                        rate = total_written / max(1, time.time() - t0)
                        print(f"processed {total_written:,} docs  ({rate:.0f}/s, {time.time()-t0:.0f}s)", flush=True)
                if total_written >= args.max_docs:
                    break
            if in_flight == 0 and total_written >= args.max_docs:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for _ in procs:
            work_q.put(None)
        for p in procs:
            p.join()
    flush()
    print(f"DONE: {total_written} docs -> {args.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
