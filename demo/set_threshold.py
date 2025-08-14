import json
from pathlib import Path

models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

with open(models_dir / "threshold.json", "w", encoding="utf-8") as f:
    json.dump({"threshold": 0.01}, f, indent=2)
print("Wrote demo/models/threshold.json with threshold=0.01")
