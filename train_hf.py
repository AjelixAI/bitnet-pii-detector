#!/usr/bin/env python3
"""Piiranha replication — canonical HF Trainer recipe, pinned to Piiranha's stack:
transformers==4.44.2, torch==2.4.1+cu121, datasets==3.0.0, tokenizers==0.19.1.

No custom training loop. Standard token-classification fine-tune exactly as in
HF docs / Piiranha's model card:
  - mdeberta-v3-base, I-only label scheme (17 PII types + O)
  - fp16 Native AMP (Trainer's GradScaler skips inf/NaN steps automatically)
  - lr 5e-5, batch 128, 5 epochs, warmup 5%, linear decay, Adam (0.9, 0.999) eps 1e-8
  - seqeval f1 every 500 steps, load_best_model_at_end
"""
import collections

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL = "/root/models/mdeberta-v3-base"
OUT = "/root/piiranha_hf"
DATA_TRAIN = "/root/data/pii400k_train.parquet"
DATA_VAL = "/root/data/pii400k_val.parquet"
SEQ = 256

PII_TYPES = ["ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATEOFBIRTH",
             "DRIVERLICENSENUM", "EMAIL", "GIVENNAME", "IDCARDNUM", "PASSWORD",
             "SOCIALNUM", "STREET", "SURNAME", "TAXNUM", "TELEPHONENUM",
             "USERNAME", "ZIPCODE"]
LABEL_NAMES = [f"I-{t}" for t in PII_TYPES] + ["O"]
LABEL_MAP = {l: i for i, l in enumerate(LABEL_NAMES)}
O_IDX = LABEL_MAP["O"]
PII_TYPE_SET = set(PII_TYPES)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    def load_clean(path):
        # strip datasets-4.x metadata so datasets 3.0.0 infers plain features
        tbl = pq.read_table(path)
        return Dataset(tbl.replace_schema_metadata({}))

    ds = DatasetDict({"train": load_clean(DATA_TRAIN),
                      "validation": load_clean(DATA_VAL)})

    def prep(batch):
        enc = tok(batch["source_text"], truncation=True, max_length=SEQ,
                  return_offsets_mapping=True)
        all_labels = []
        for offs, mask in zip(enc["offset_mapping"], batch["privacy_mask"]):
            spans = [(s["start"], s["end"], s["label"]) for s in mask
                     if s["label"] in PII_TYPE_SET]
            lab = []
            for a, b in offs:
                if a == b:                       # special tokens
                    lab.append(-100)
                    continue
                tag = O_IDX
                for s, e, t in spans:
                    if not (b <= s or a >= e):   # char overlap
                        tag = LABEL_MAP["I-" + t]
                        break
                lab.append(tag)
            all_labels.append(lab)
        enc["labels"] = all_labels
        del enc["offset_mapping"]
        return enc

    cols = ds["train"].column_names
    train_ds = ds["train"].map(prep, batched=True, batch_size=1000, num_proc=8,
                               remove_columns=cols, desc="tokenize train")

    # stratified eval subset: 1200 rows x 6 languages
    val = ds["validation"]
    by_lang = collections.defaultdict(list)
    for i, lg in enumerate(val["language"]):
        by_lang[lg].append(i)
    sel = []
    for lg in sorted(by_lang):
        sel += by_lang[lg][:1200]
    eval_ds = val.select(sel).map(prep, batched=True, batch_size=1000,
                                  remove_columns=cols, desc="tokenize eval")
    print(f"train: {len(train_ds)}  eval: {len(eval_ds)} ({len(by_lang)} langs)", flush=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(LABEL_NAMES),
        id2label=dict(enumerate(LABEL_NAMES)), label2id=LABEL_MAP)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, -1)
        true_pred, true_lab = [], []
        for pr, lb in zip(preds, p.label_ids):
            tp_, tl_ = [], []
            for pi, li in zip(pr, lb):
                if li == -100:
                    continue
                tp_.append(LABEL_NAMES[pi])
                tl_.append(LABEL_NAMES[li])
            true_pred.append(tp_)
            true_lab.append(tl_)
        return {"precision": precision_score(true_lab, true_pred),
                "recall": recall_score(true_lab, true_pred),
                "f1": f1_score(true_lab, true_pred)}

    args = TrainingArguments(
        output_dir=OUT,
        learning_rate=5e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        weight_decay=0.0,
        fp16=True,                       # Piiranha: Native AMP
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        seed=42,
        dataloader_num_workers=4,
        report_to=["wandb"],
        run_name="piiranha-hf-pinned",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    m = trainer.evaluate()
    print(f"FINAL: {m}", flush=True)
    trainer.save_model(OUT + "/best")
    tok.save_pretrained(OUT + "/best")


if __name__ == "__main__":
    main()
