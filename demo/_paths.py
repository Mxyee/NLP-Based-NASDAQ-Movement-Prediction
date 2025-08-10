
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"

RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
