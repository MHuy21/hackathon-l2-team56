import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_train_notebook", ROOT / "scripts" / "colab" / "make_train_notebook.py")
mtn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtn)


def test_only_parameter_cell_changes_and_values_are_set(tmp_path):
    original = json.loads((ROOT / "notebooks" / "train_and_export.ipynb").read_text(encoding="utf-8"))
    nb = mtn.build("r00", 40, 16, "/content/drive/MyDrive/team56")
    i = mtn.param_cell_index(nb)
    changed = [k for k, (a, b) in enumerate(zip(original["cells"], nb["cells"])) if a != b]
    assert changed == [i]
    assert nb["metadata"] == original["metadata"]
    g = {"IN_COLAB": True, "os": __import__("os")}
    exec("".join(nb["cells"][i]["source"]), g)
    assert g["TEAM_ID"] == 61
    assert g["DATASET_PATH"] == "/content/drive/MyDrive/team56/du_lieu_r00.zip"
    assert g["EPOCHS"] == 40 and g["BATCH"] == 16
    assert g["DRIVE_OUTPUT_DIR"] == "/content/drive/MyDrive/team56/outputs/r00"
    assert g["PARAMS"]["EPOCHS"] == 40  # recorded in run_report.json
    g["offer_download"](["/content/work/submission/submission.zip"])  # headless: prints, never opens a browser


def test_cli_writes_runs_copy(tmp_path):
    out = tmp_path / "train_r01.ipynb"
    assert mtn.main(["--round", "r01", "--epochs", "30", "--out", str(out)]) == 0
    nb = json.loads(out.read_text(encoding="utf-8"))
    assert "du_lieu_r01.zip" in "".join(nb["cells"][mtn.param_cell_index(nb)]["source"])
