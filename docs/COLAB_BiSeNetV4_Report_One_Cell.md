# Colab one-cell collapse diagnostics for BiSeNet V4

Copy the whole cell below into one Google Colab code cell.

This is **Direct Execution Mode**:

- no repo clone
- no pull
- no checkpoint download
- no inline report logic
- only validate local / mounted paths and run `tools/eval_bisenet_diagnostics.py`

Before running, make sure the repo folder in Colab already contains the latest
`tools/eval_bisenet_diagnostics.py`.

```python
# ============================================================
# BiSeNet V4 one-cell Colab report (Direct Execution Mode)
# - mount Drive
# - validate local / mounted paths
# - run tools/eval_bisenet_diagnostics.py
# - export collapse / sink-class / FG-BG diagnostics
# ============================================================

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


# ----------------------------
# 1) Path configuration
# ----------------------------
# Repo already available in Colab. This cell does not clone or pull.
REPO_DIR = Path("/content/SemanticSegmentation-BiSeNet-FoodSeg103")

# FoodSeg103 root containing:
# class_mapping.json, train/img, train/mask, test/img, test/mask.
DATA_ROOT = Path("/content/data/foodseg103-full")

# Static checkpoint path on Google Drive.
CKPT_PATH = Path(
    "/content/drive/MyDrive/[PROJECT][COMPUTER-VISION]/BISENET-CONFIG/"
    "bisenet_v4[all-change]/best_miou.pth"
)

# Base output folder on Google Drive.
REPORT_DIR = Path(
    "/content/drive/MyDrive/[PROJECT][COMPUTER-VISION]/BISENET-CONFIG/"
    "bisenet_v4[all-change]"
)
OUTPUT_DIR = REPORT_DIR / "diagnostics" / "collapse_diagnostics"


# ----------------------------
# 2) Runtime configuration
# ----------------------------
SPLIT = "test"
EVAL_SIZE = "768x768"     # use "none" for original image size
BATCH_SIZE = 1
NUM_WORKERS = 2
TOP_K = 20
EXAMPLES_PER_GROUP = 8

# Smoke test:
# MAX_ITEMS = 50
# SKIP_TRAIN_PRESENCE = True
# NO_PLOTS = True
#
# Full run:
# MAX_ITEMS = None
# SKIP_TRAIN_PRESENCE = False
# NO_PLOTS = False
MAX_ITEMS = None
SKIP_TRAIN_PRESENCE = False
NO_PLOTS = False

# Keep this False to preserve Direct Execution Mode: no network install.
# If dependency validation fails, run the printed pip command manually in a
# separate Colab cell, then rerun this cell.
AUTO_INSTALL_MISSING = False


# ----------------------------
# 3) Helpers
# ----------------------------
def require_path(path: Path, label: str, is_dir: bool | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if is_dir is True and not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    if is_dir is False and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def print_config() -> None:
    print("Runtime configuration")
    print("-" * 80)
    print(f"REPO_DIR            : {REPO_DIR}")
    print(f"DATA_ROOT           : {DATA_ROOT}")
    print(f"CKPT_PATH           : {CKPT_PATH}")
    print(f"REPORT_DIR          : {REPORT_DIR}")
    print(f"OUTPUT_DIR          : {OUTPUT_DIR}")
    print(f"SPLIT               : {SPLIT}")
    print(f"EVAL_SIZE           : {EVAL_SIZE}")
    print(f"BATCH_SIZE          : {BATCH_SIZE}")
    print(f"NUM_WORKERS         : {NUM_WORKERS}")
    print(f"MAX_ITEMS           : {MAX_ITEMS}")
    print(f"SKIP_TRAIN_PRESENCE : {SKIP_TRAIN_PRESENCE}")
    print(f"NO_PLOTS            : {NO_PLOTS}")
    print("-" * 80)


def build_command(script_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--ckpt",
        str(CKPT_PATH),
        "--data-root",
        str(DATA_ROOT),
        "--work-dir",
        str(REPORT_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
        "--split",
        SPLIT,
        "--batch-size",
        str(BATCH_SIZE),
        "--num-workers",
        str(NUM_WORKERS),
        "--eval-size",
        str(EVAL_SIZE),
        "--top-k",
        str(TOP_K),
        "--examples-per-group",
        str(EXAMPLES_PER_GROUP),
    ]
    if MAX_ITEMS is not None:
        cmd.extend(["--max-items", str(MAX_ITEMS)])
    if SKIP_TRAIN_PRESENCE:
        cmd.append("--skip-train-presence")
    if NO_PLOTS:
        cmd.append("--no-plots")
    return cmd


def validate_python_modules() -> None:
    required = {
        "numpy": "numpy",
        "torch": "torch",
        "matplotlib": "matplotlib",
        "PIL": "pillow",
        "tqdm": "tqdm",
        "cv2": "opencv-python",
        "albumentations": "albumentations",
    }
    missing = [pip_name for module, pip_name in required.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REPO_DIR / "requirements.txt"),
        *missing,
    ]
    print("[ERROR] Missing Python modules required by eval_bisenet_diagnostics.py:")
    for package in missing:
        print(f" - {package}")
    print("\nRun this in a separate Colab cell, then rerun this cell:")
    print(" ".join(install_cmd))

    if AUTO_INSTALL_MISSING:
        subprocess.run(install_cmd, cwd=REPO_DIR, check=True)
        return

    raise ModuleNotFoundError("Missing required Python modules. Install them first, then rerun this cell.")


def run_with_debug(cmd: list[str]) -> None:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_DIR}:{old_pythonpath}" if old_pythonpath else str(REPO_DIR)

    result = subprocess.run(
        cmd,
        cwd=REPO_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.stdout:
        print("\n--- diagnostics stdout ---")
        print(result.stdout)
    if result.stderr:
        print("\n--- diagnostics stderr ---")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "eval_bisenet_diagnostics.py failed. "
            "Read the diagnostics stderr/stdout above for the real error."
        )


# ----------------------------
# 4) Mount Drive
# ----------------------------
try:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print(f"[WARN] Drive mount skipped or failed: {exc}")


# ----------------------------
# 5) Strict path validation
# ----------------------------
script_path = REPO_DIR / "tools" / "eval_bisenet_diagnostics.py"

require_path(REPO_DIR, "Repo directory", is_dir=True)
require_path(script_path, "Diagnostic script", is_dir=False)
require_path(REPO_DIR / "requirements.txt", "requirements.txt", is_dir=False)
require_path(CKPT_PATH, "Checkpoint", is_dir=False)
require_path(DATA_ROOT, "DATA_ROOT", is_dir=True)
require_path(DATA_ROOT / "class_mapping.json", "class_mapping.json", is_dir=False)
require_path(DATA_ROOT / "test" / "img", "test/img", is_dir=True)
require_path(DATA_ROOT / "test" / "mask", "test/mask", is_dir=True)

if not SKIP_TRAIN_PRESENCE:
    require_path(DATA_ROOT / "train" / "img", "train/img", is_dir=True)
    require_path(DATA_ROOT / "train" / "mask", "train/mask", is_dir=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
validate_python_modules()


# ----------------------------
# 6) Run diagnostics
# ----------------------------
print_config()
os.chdir(REPO_DIR)
cmd = build_command(script_path)

print("Command")
print("-" * 80)
print(" ".join(cmd))
print("-" * 80)

run_with_debug(cmd)


# ----------------------------
# 7) Output check
# ----------------------------
expected_outputs = [
    "summary.txt",
    "summary.json",
    "collapse_summary.json",
    "binary_fg_bg_metrics.json",
    "class_diagnostics.csv",
    "sink_class_analysis.csv",
    "per_image_diagnostics.csv",
    "prediction_distribution.csv",
]

print("\nGenerated report")
print("-" * 80)
print(f"Report directory: {OUTPUT_DIR}")
for name in expected_outputs:
    path = OUTPUT_DIR / name
    status = "OK" if path.exists() else "MISSING"
    print(f"[{status}] {path}")
print(f"[INFO] Plots directory: {OUTPUT_DIR / 'plots'}")
```
