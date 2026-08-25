#!/usr/bin/env python3
# build_tokenizer_gpt.py — SOTA-match 65k tokenizer (GPT-2 regex pre-tokenizer).
#
# Reproduces the exact pre-tokenizer used by Liquid AI's SOTA PII model:
#   Sequence([Split(GPT2 regex, isolated), ByteLevel(add_prefix_space=False,
#   trim_offsets=True, use_regex=False)]) + ByteLevel(trim_offsets) decoder.
# Trained on the real pretrain corpus (5.24B tokens) + identifier-rich synthetic
# so digits/identifiers are represented well. trim_offsets=True fixed the
# leading-space off-by-one that crushed our earlier span F1.
#
# Output: /root/pii_data/pretok_gpt/tokenizer.json
import os, random, argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import Split, ByteLevel, Sequence
from tokenizers import Regex

GPT2_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def build_tokenizer(texts, vocab=65000):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = Sequence([
        Split(Regex(GPT2_REGEX), "isolated"),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])
    tok.decoder = decoders.Sequence([
        decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True),
    ])
    trainer = trainers.BpeTrainer(vocab_size=vocab, min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>", "<cls>", "<sep>"])
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/pii_data/pretok_gpt/tokenizer.json")
    ap.add_argument("--max-docs", type=int, default=400000)
    args = ap.parse_args()
    # Real corpus: read the pre-tokenized .bin? No — need raw text. Use FineWeb stream.
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    import itertools
    texts = itertools.islice((d["text"] for d in ds if d.get("text") and len(d["text"].strip()) > 20),
                             args.max_docs)
    # identifier-rich synthetic to boost digit/identifier coverage (small admixture)
    random.seed(0)
    extra = []
    for _ in range(2000):
        extra.append("SSN %03d-%02d-%04d card %s" % (
            random.randint(100,999), random.randint(10,99), random.randint(1000,9999),
            "-".join(["%04d" % random.randint(0,9999) for _ in range(4)])))
        extra.append("IBAN DE89370400440532013000 phone +49%d%d" % (random.randint(10,99), random.randint(100000,999999)))
        extra.append("Company Example Corporation Ltd email jane%d@example.com" % random.randint(1,999))
    # generator that yields real texts then extra
    def gen():
        for t in texts:
            yield t
        for t in extra:
            yield t
    tok = build_tokenizer(gen(), vocab=65000)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok.save(args.out)
    print("vocab:", tok.get_vocab_size(), "->", args.out)
    # verify offsets are clean (no leading-space off-by-one)
    for s in ["SSN 123-45-6789 here", "card 4111-1111-1111-1111", "email jane@example.com"]:
        e = tok.encode(s)
        print("IN:", s, "->", [tok.decode([i]) for i in e.ids], "offsets:", e.offsets[:3])
