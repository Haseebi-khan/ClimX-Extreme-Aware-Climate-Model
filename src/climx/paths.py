"""Project paths. Override with the environment variables CLIMX_DATA and CLIMX_WORK."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CLIMX_DATA", ROOT / "data"))
WORK = Path(os.environ.get("CLIMX_WORK", ROOT / "work"))
HF_FULL = DATA / "hf" / "full"
KAGGLE_LITE = DATA / "kaggle" / "archive"
SAMPLE_SUBMISSION = DATA / "kaggle" / "sample_submission.parquet"
OFFICIAL_REPO = ROOT / "external" / "ClimX-official"

SCENARIOS = ("historical", "ssp126", "ssp370", "ssp585")
TEST_SCENARIO = "ssp245"
INDEX_NAMES = ("TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
               "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII")
NLAT, NLON = 192, 288
