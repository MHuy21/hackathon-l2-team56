"""[runs ON the Colab VM via `colab exec -f`, after jobs/prepare.py] Draft labels for the VN training images.

Parameters (``colab exec --env KEY=VALUE``):
    SOURCES       comma list of sources to draft    (default: roboflow_kien,kaggle_day)
    KAGGLE_SAMPLE seeded sample size for kaggle_day (default: 1000)
    USE_SAM       1 = refine masks with SAM 2       (default: 1)
    DRIVE_DIR     team folder on Drive              (default: /content/drive/MyDrive/team56)

Writes <DRIVE_DIR>/drafts/roboflow_kien and <DRIVE_DIR>/drafts/kaggle_vn (resumable: rerun to continue), then
prints per-class totals and runs the QA validator on each draft folder.
"""
import csv
import os
import subprocess
import sys

DRIVE = os.environ.get("DRIVE_DIR", "/content/drive/MyDrive/team56")
SOURCES = [s.strip() for s in os.environ.get("SOURCES", "roboflow_kien,kaggle_day").split(",") if s.strip()]
SAMPLE = int(os.environ.get("KAGGLE_SAMPLE", "1000"))
USE_SAM = os.environ.get("USE_SAM", "1") == "1"
REPO = "/content/team56"
RAW = "/content/vn_raw"
OUT_NAME = {"roboflow_kien": "roboflow_kien", "kaggle_day": "kaggle_vn"}


def run_live(cmd):
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")


for src in SOURCES:
    if src not in OUT_NAME:
        raise SystemExit(f"unknown source {src!r}; expected one of {list(OUT_NAME)}")
    out = f"{DRIVE}/drafts/{OUT_NAME[src]}"
    cmd = [sys.executable, f"{REPO}/scripts/label/autolabel.py", "--images", f"{RAW}/{src}", "--out", out]
    if USE_SAM:
        cmd += ["--sam", "sam2.1_b.pt"]
    if src == "kaggle_day" and SAMPLE:
        cmd += ["--sample", str(SAMPLE)]
    run_live(cmd)

print("\n=== summary ===", flush=True)
for src in SOURCES:
    out = f"{DRIVE}/drafts/{OUT_NAME[src]}"
    with open(f"{out}/autolabel_report.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_lbl = len(os.listdir(f"{out}/labels"))
    n_img = len(os.listdir(f"{out}/images"))
    print(f"\n{OUT_NAME[src]}: {len(rows)} in report | {n_img} images | {n_lbl} label files", flush=True)
    for c in [c for c in rows[0] if c.startswith("n_")]:
        print(f"  {c[2:]:<14} {sum(int(r[c]) for r in rows):>6}", flush=True)
    run_live([sys.executable, f"{REPO}/scripts/qa/validate_labels.py", out, "--max-details", "5"])
