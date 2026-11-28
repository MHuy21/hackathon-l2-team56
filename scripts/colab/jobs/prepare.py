"""[runs ON the Colab VM via `colab exec -f`] Get code, dependencies and raw VN images ready on the runtime.

Prerequisites (done by scripts/colab/pipeline.sh): GPU session created, Google Drive mounted at /content/drive.
Parameters (``colab exec --env KEY=VALUE``):
    REPO_URL   git URL of hackathon-l2-team56            (default: the team repo on GitHub)
    REPO_REF   branch / commit to check out              (default: main)
    DRIVE_DIR  team folder on Drive                      (default: /content/drive/MyDrive/team56)

Idempotent: rerunning pulls the latest code and skips archives that are already extracted.
Night-time Kaggle frames are left out of the "kaggle_day" folder: B found detection boxes baked into the pixels of
some night groups of that dataset (golden_val_150/selection_report.json).
"""
import glob
import os
import subprocess
import sys
import zipfile

REPO_URL = os.environ.get("REPO_URL", "https://github.com/MHuy21/hackathon-l2-team56.git")
REPO_REF = os.environ.get("REPO_REF", "main")
DRIVE = os.environ.get("DRIVE_DIR", "/content/drive/MyDrive/team56")
REPO = "/content/team56"
RAW = "/content/vn_raw"
ARCHIVES = {"roboflow_kien": "roboflow_790.zip", "kaggle_vn": "kaggle_3000.zip", "golden_val_150": "golden_val_150.zip"}
IMG_EXT = (".jpg", ".jpeg", ".png")


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def images_under(folder):
    return sorted(p for p in glob.glob(f"{folder}/**/*", recursive=True) if p.lower().endswith(IMG_EXT))


if not os.path.isdir("/content/drive/MyDrive"):
    raise SystemExit("Google Drive is not mounted: run `scripts/colab/pipeline.sh mount` first")

# 1. code at the requested ref
if not os.path.isdir(f"{REPO}/.git"):
    sh(["git", "clone", "-q", REPO_URL, REPO])
sh(["git", "-C", REPO, "fetch", "-q", "--all"])
sh(["git", "-C", REPO, "checkout", "-q", REPO_REF])
on_branch = subprocess.run(["git", "-C", REPO, "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 0
if on_branch:  # a commit hash is already exact; a branch may have moved since the clone
    sh(["git", "-C", REPO, "pull", "-q", "--ff-only"])
print(subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout, flush=True)

# 2. dependencies (Ultralytics pinned like the official notebook)
sh([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.214", "pycocotools", "shapely", "imagehash", "pyyaml"])

# 3. raw images from Drive
for name, archive in ARCHIVES.items():
    dest, src = f"{RAW}/{name}", f"{DRIVE}/{archive}"
    if images_under(dest):
        print(f"{name}: already extracted", flush=True)
        continue
    if not os.path.isfile(src):
        raise SystemExit(f"missing {src} on Drive")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)
    print(f"{name}: extracted {src}", flush=True)

# 4. day-only Kaggle view (symlinks), names must stay unique
day = f"{RAW}/kaggle_day"
os.makedirs(day, exist_ok=True)
seen, skipped_night, clashes = set(), 0, 0
for p in images_under(f"{RAW}/kaggle_vn"):
    if "night" in os.path.relpath(p, f"{RAW}/kaggle_vn").lower():
        skipped_night += 1
        continue
    name = os.path.basename(p)
    if name in seen:
        clashes += 1
        continue
    seen.add(name)
    link = f"{day}/{name}"
    if not os.path.lexists(link):
        os.symlink(p, link)

for name in ARCHIVES:
    print(f"{name:<16} {len(images_under(f'{RAW}/{name}')):>6} images", flush=True)
print(f"kaggle_day       {len(seen):>6} images (left out: {skipped_night} night-time, {clashes} duplicate names)", flush=True)
excl = glob.glob(f"{RAW}/golden_val_150/**/exclude_from_train_same_frames.csv", recursive=True)
print("golden-val exclude list:", excl[0] if excl else "NOT FOUND", flush=True)

import torch  # noqa: E402

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (drafting will be very slow)", flush=True)
