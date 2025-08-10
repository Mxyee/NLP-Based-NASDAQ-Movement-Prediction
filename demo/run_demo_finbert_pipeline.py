
# demo/run_demo_finbert_pipeline.py
import sys, datetime as dt, pandas as pd
from pathlib import Path

# Ensure demo data folders exist and get absolute paths
from _paths import RAW, PROC

# Make news_pipeline import robust: try demo/, then project root
HERE = Path(__file__).resolve().parent            # .../demo
ROOT = HERE.parent                                 # project root
CANDIDATES = [HERE, ROOT]
for p in CANDIDATES:
    if (p / "news_pipeline.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

try:
    from news_pipeline import run_news_pipeline, NEWYORK
except Exception as e:
    raise SystemExit(f"Could not import news_pipeline.py from {CANDIDATES}. Error: {e}")

def main():
    # Use ET 09:29 cutoff by default
    cutoff = dt.datetime.now(NEWYORK).replace(hour=9, minute=29, second=59, microsecond=0)
    metrics, detailed = run_news_pipeline(cutoff_et=cutoff, batch_size=8)

    # Save detailed and metrics to absolute paths; ensure dirs exist
    PROC.mkdir(parents=True, exist_ok=True)
    out_det = (PROC / "finbert_news_scored.csv").resolve()
    out_met = (PROC / "finbert_daily_metrics.csv").resolve()

    detailed.to_csv(out_det, index=False)
    pd.DataFrame([metrics]).to_csv(out_met, index=False)

    print(f"[B1] Saved FinBERT-scored news → {out_det} ({len(detailed)} rows)")
    print(f"[B2] Saved daily metrics → {out_met}: {metrics}")

    # Console summary
    show_cols = [c for c in ["pub_time_et","title","prob_pos","prob_neg","w"] if c in detailed.columns]
    if show_cols:
        print("\n[B] Top rows:")
        print(detailed.sort_values("w", ascending=False).head(5)[show_cols])

if __name__ == "__main__":
    main()
