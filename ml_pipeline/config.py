import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Configurable dataset root via environment variable AQUAGUARD_DATASET_ROOT
# Can be overridden by the CLI argument --dataset-root in run_audit.py
AQUAGUARD_DATASET_ROOT = os.getenv("AQUAGUARD_DATASET_ROOT", "")

# Report outputs
REPORT_JSON_PATH = os.getenv("REPORT_JSON_PATH", str(BASE_DIR / "artifacts" / "dataset_report.json"))
REPORT_MD_PATH = os.getenv("REPORT_MD_PATH", str(BASE_DIR / "artifacts" / "dataset_report.md"))

def get_dataset_config() -> dict:
    """
    Returns the current dataset configuration.
    Default dataset_root is retrieved from the AQUAGUARD_DATASET_ROOT env variable.
    """
    return {
        "dataset_root": AQUAGUARD_DATASET_ROOT,
        "report_json_path": REPORT_JSON_PATH,
        "report_md_path": REPORT_MD_PATH,
    }
