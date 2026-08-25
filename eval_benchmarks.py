#!/usr/bin/env python3
# eval_benchmarks.py — head-to-head vs LiquidAI's published numbers on MAPA + TAB.
# Loads the benchmark slots from their leaderboard and runs our model on the same data.
import argparse, json, subprocess, os
import torch

# LiquidAI's reported PII-detector numbers (from their model card)
REFERENCE = {
    "SPY":        {"lfm2.5": 0.428, "next_best": 0.280},   # SauerkrautLM GLiNER
    "Gretel":     {"lfm2.5": 0.880, "next_best": 0.663},
    "TAB":        {"lfm2.5": 0.867, "next_best": 0.685},
    "ai4privacy": {"lfm2.5": 0.715, "next_best": 0.946},   # Piiranha (in-distribution)
    "Nemotron":   {"lfm2.5": 0.855, "next_best": 0.918},   # OpenMed (in-distribution)
    "MAPA":       {"lfm2.5": 0.236, "next_best": 0.416},   # their weakest — our wedge
}
TARGETS = {"MAPA": 0.42, "TAB": 0.87, "ai4privacy": 0.75, "SPY": 0.44, "Nemotron": 0.86}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/pii_corpus/eurobert_pii_eu.pt")
    ap.add_argument("--benchdir", default="/scratch/pii_corpus/benchmarks")
    args = ap.parse_args()
    print("Reference (LiquidAI card) vs targets-to-win:")
    for b, ref in REFERENCE.items():
        tgt = TARGETS.get(b, ref["lfm2.5"])
        print(f"  {b:11s} lfm2.5={ref['lfm2.5']:.3f}  next_best_external={ref['next_best']:.3f}  target={tgt:.2f}")
    print("\nBenchmarks dir:", args.benchdir)
    for name in ("mapa", "tab", "spy"):
        p = os.path.join(args.benchdir, name)
        print(f"  {name}: {'PRESENT' if os.path.exists(p) else 'TODO download'}")

if __name__ == "__main__":
    main()
