"""[runs ON the Colab VM via `colab exec -f`, after jobs/autolabel_vn.py] Build a round's dataset zip.

Parameters (``colab exec --env KEY=VALUE``):
    ROUND      recipe name in configs/datasets/   (default: r00)
    DRIVE_DIR  team folder on Drive               (default: /content/drive/MyDrive/team56)

Lays out the inputs where the recipe expects them (repo-relative data/interim/...), runs the builder's checks, then
writes du_lieu_<round>.zip and copies it plus its manifests to <DRIVE_DIR>/ (zip) and <DRIVE_DIR>/manifests/.
Stops without writing anything when the builder reports an error.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

ROUND = os.environ.get("ROUND", "r00")
DRIVE = os.environ.get("DRIVE_DIR", "/content/drive/MyDrive/team56")
REPO = "/content/team56"
INTERIM = f"{REPO}/data/interim"


def run_live(cmd):
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")


def fresh_copy(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"copied {src} -> {dst} ({len(os.listdir(os.path.join(dst, 'images')))} images)", flush=True)


# COCO (converted on Colab, PASSed QA): unzip once
coco = f"{INTERIM}/coco"
if not glob.glob(f"{coco}/images/*"):
    os.makedirs(coco, exist_ok=True)
    with zipfile.ZipFile(f"{DRIVE}/coco_r00.zip") as zf:
        zf.extractall(coco)
print(f"coco: {len(glob.glob(f'{coco}/images/*'))} images", flush=True)

# VN drafts: local copies (reading thousands of small files through the Drive mount is slow)
for name in ("roboflow_kien", "kaggle_vn"):
    fresh_copy(f"{DRIVE}/drafts/{name}", f"{INTERIM}/drafts/{name}")

# golden val: exclusion list for every round, the corrected labels for R1+
excl = glob.glob("/content/vn_raw/golden_val_150/**/exclude_from_train_same_frames.csv", recursive=True)
if not excl:
    raise SystemExit("golden-val exclude list not found: run jobs/prepare.py first")
os.makedirs(f"{REPO}/manifests/golden_val_150", exist_ok=True)
shutil.copyfile(excl[0], f"{REPO}/manifests/golden_val_150/exclude_from_train_same_frames.csv")
if ROUND != "r00":
    gv = f"{DRIVE}/golden_val"
    if not os.path.isdir(f"{gv}/images"):
        raise SystemExit(f"{gv}/images missing: B uploads the corrected golden val (CVAT export via from_cvat.py) there")
    fresh_copy(gv, f"{INTERIM}/golden_val")

recipe = f"configs/datasets/{ROUND}.yaml"
run_live([sys.executable, "scripts/data/build_dataset.py", recipe, "--check"])
run_live([sys.executable, "scripts/data/build_dataset.py", recipe])

name = f"du_lieu_{ROUND}"
os.makedirs(f"{DRIVE}/manifests", exist_ok=True)
shutil.copyfile(f"{REPO}/data/processed/{name}.zip", f"{DRIVE}/{name}.zip")
for ext in ("csv", "json"):
    shutil.copyfile(f"{REPO}/manifests/{name}.{ext}", f"{DRIVE}/manifests/{name}.{ext}")
print(f"\nzip on Drive: {DRIVE}/{name}.zip  (DATASET_PATH for the training notebook)", flush=True)
print(f"manifests:    {DRIVE}/manifests/{name}.csv/.json  and on the VM in {REPO}/manifests/", flush=True)
