"""Write a ready-to-run copy of the official training notebook for one round (only the team-parameter cell changes).

    python scripts/colab/make_train_notebook.py --round r00 --epochs 40
    -> notebooks/runs/train_r00.ipynb  (DATASET_PATH = <Drive>/du_lieu_r00.zip, outputs to <Drive>/outputs/r00)

The copy is what `scripts/colab/pipeline.sh train` sends to the Colab VM with `colab exec -f`; the executed copy
comes back as notebooks/runs/train_r00_output.ipynb. Everything outside the parameter cell (export contract,
self-check, scoring) stays byte-identical to notebooks/train_and_export.ipynb, and that is verified here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "notebooks" / "train_and_export.ipynb"
PARAM_TITLE = "#@title Tham số của đội"
HEADLESS_OVERRIDE = '''# [colab-cli] headless run: print where the files are instead of opening a browser download
def offer_download(paths):
    for p in paths:
        print(f"Tệp: {p}")
'''


def param_cell_index(nb: dict) -> int:
    idx = [i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith(PARAM_TITLE)]
    if len(idx) != 1:
        raise SystemExit(f"expected exactly one parameter cell in {SOURCE}, found {len(idx)}")
    return idx[0]


def set_assignment(src: str, name: str, value: str) -> str:
    """Replace the single-line assignment `NAME = ...` (keeping any trailing `#@param` comment)."""
    pattern = re.compile(rf"^{name} = [^\n#]*(#[^\n]*)?$", re.M)
    if len(pattern.findall(src)) != 1:
        raise SystemExit(f"parameter {name} not found exactly once in the parameter cell")
    return pattern.sub(lambda m: f"{name} = {value}" + (f"  {m.group(1)}" if m.group(1) else ""), src)


def build(round_name: str, epochs: int, batch: int, drive_dir: str) -> dict:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    i = param_cell_index(nb)
    src = "".join(nb["cells"][i]["source"])
    if "TEAM_ID = 61" not in src:
        raise SystemExit("TEAM_ID is not 61 in the source notebook")
    src = set_assignment(src, "DATASET_PATH", json.dumps(f"{drive_dir}/du_lieu_{round_name}.zip"))
    src = set_assignment(src, "EPOCHS", str(epochs))
    src = set_assignment(src, "BATCH", str(batch))
    # DRIVE_OUTPUT_DIR is a two-line expression in the team copy: replace the whole statement
    m = re.search(r"^DRIVE_OUTPUT_DIR = \(.*?\)\n", src, re.M | re.S)
    if not m:
        raise SystemExit("DRIVE_OUTPUT_DIR statement not found in the parameter cell")
    src = src[: m.start()] + f"DRIVE_OUTPUT_DIR = {json.dumps(f'{drive_dir}/outputs/{round_name}')}\n" + src[m.end():]
    # Run headless by colab-cli: no browser tab is attached to the kernel, and google.colab.files.download (used by
    # offer_download in Step 6) would wait for one forever. This cell runs after the library cells, so redefine it.
    src = src.rstrip("\n") + "\n\n" + HEADLESS_OVERRIDE
    nb["cells"][i]["source"] = src.splitlines(keepends=True)
    return nb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, help="dataset round, e.g. r00 (uses du_lieu_<round>.zip)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--drive-dir", default="/content/drive/MyDrive/team56")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    nb = build(args.round, args.epochs, args.batch, args.drive_dir)
    out = args.out or ROOT / "notebooks" / "runs" / f"train_{args.round}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
