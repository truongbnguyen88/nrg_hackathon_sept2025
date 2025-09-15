#!/usr/bin/env python3
import argparse, sys, pandas as pd

# Both schemas supported
PHASE1_COLS = ["site_id","timestamp","demand_response_event"]
PHASE2_COLS = ["site_id","timestamp","demand_response_capacity"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="submissions/predictions.csv")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERROR] Could not read {args.input}: {e}"); sys.exit(1)

    cols = df.columns.tolist()
    if all(c in cols for c in PHASE1_COLS):
        expected = PHASE1_COLS
        print("[INFO] Detected Phase 1 submission")
    elif all(c in cols for c in PHASE2_COLS):
        expected = PHASE2_COLS
        print("[INFO] Detected Phase 2 submission")
    else:
        print(f"[ERROR] Columns must match Phase 1 {PHASE1_COLS} or Phase 2 {PHASE2_COLS}")
        sys.exit(2)

    missing = [c for c in expected if c not in cols]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        sys.exit(3)

    if df.isna().any().any():
        print("[WARN] Found NaNs.")

    print("[OK] Submission looks valid.")
    print("[OK] Shape:", df.shape)
    print("[OK] Columns:", cols)

if __name__ == "__main__":
    main()
