import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="submissions/predictions.csv")
args = parser.parse_args()

# Dummy baseline submission (predicts 0 for all)
df = pd.DataFrame({
    "site_id": [0],
    "timestamp": ["2025-01-01 00:00:00"],
    "demand_response_event": [0]  # Change column for Phase 2
})
df.to_csv(args.output, index=False)
print(f"Wrote {args.output}")
