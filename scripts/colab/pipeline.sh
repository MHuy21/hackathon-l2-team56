#!/usr/bin/env bash
# Drive the team's Colab work from a terminal with google-colab-cli (Linux / macOS / WSL; see scripts/colab/README.md).
#
#   scripts/colab/pipeline.sh new                 # T4 GPU session "team56"
#   scripts/colab/pipeline.sh mount               # mount Google Drive (approve once in the browser)
#   scripts/colab/pipeline.sh prepare             # code + deps + raw VN images on the VM
#   scripts/colab/pipeline.sh autolabel           # draft labels: Roboflow 790 + Kaggle day sample (~70 min)
#   scripts/colab/pipeline.sh build r00           # du_lieu_r00.zip -> Drive, manifests -> local repo
#   scripts/colab/pipeline.sh train r00 40        # official notebook, 40 epochs (~2 h); executed copy saved locally
#   scripts/colab/pipeline.sh fetch r00           # submission.zip + run_report.json -> experiments/round_00/
#   scripts/colab/pipeline.sh stop                # release the VM
#   scripts/colab/pipeline.sh r0                  # all of the above in order (mount still asks for the browser)
#
# Environment: COLAB_SESSION (team56), COLAB_GPU (T4), REPO_REF (main), KAGGLE_SAMPLE (1000), SOURCES, USE_SAM (1).
set -euo pipefail

cd "$(dirname "$0")/../.."
SESSION="${COLAB_SESSION:-team56}"
GPU="${COLAB_GPU:-T4}"
REF="${REPO_REF:-main}"
# WSL / headless Linux has no unlocked OS keyring: keep the OAuth token in a user-only file instead
export PYTHON_KEYRING_BACKEND="${PYTHON_KEYRING_BACKEND:-keyrings.alt.file.PlaintextKeyring}"

job() {  # job <file> [KEY=VALUE ...]: run a local script inside the session's kernel, no idle limit
  local file="$1"; shift
  local envs=()
  for kv in "$@"; do envs+=(--env "$kv"); done
  colab exec -s "$SESSION" --timeout 0 "${envs[@]}" -f "$file"
}

round_dir() {  # r00 -> experiments/round_00
  echo "experiments/round_${1#r}"
}

step="${1:-}"; shift || true
case "$step" in
  new)
    colab new -s "$SESSION" --gpu "$GPU" ;;
  mount)
    colab drivemount -s "$SESSION" ;;
  prepare)
    job scripts/colab/jobs/prepare.py "REPO_REF=$REF" ;;
  autolabel)
    job scripts/colab/jobs/autolabel_vn.py "KAGGLE_SAMPLE=${KAGGLE_SAMPLE:-1000}" \
      "SOURCES=${SOURCES:-roboflow_kien,kaggle_day}" "USE_SAM=${USE_SAM:-1}" ;;
  build)
    round="${1:-r00}"
    job scripts/colab/jobs/build_round.py "ROUND=$round"
    mkdir -p manifests
    for ext in csv json; do
      colab download -s "$SESSION" "/content/team56/manifests/du_lieu_${round}.${ext}" "manifests/du_lieu_${round}.${ext}"
    done
    echo "manifests saved in manifests/ (commit them)" ;;
  train)
    round="${1:-r00}"; epochs="${2:-40}"
    python3 scripts/colab/make_train_notebook.py --round "$round" --epochs "$epochs"
    colab exec -s "$SESSION" --timeout 0 -f "notebooks/runs/train_${round}.ipynb"
    echo "executed notebook: notebooks/runs/train_${round}_output.ipynb" ;;
  fetch)
    round="${1:-r00}"; dir="$(round_dir "$round")"
    mkdir -p "$dir"
    colab download -s "$SESSION" /content/work/submission/submission.zip "$dir/submission.zip"
    colab download -s "$SESSION" /content/work/submission/run_report.json "$dir/run_report.json"
    echo "upload $dir/submission.zip on the competition website; commit $dir/run_report.json" ;;
  status)
    colab sessions; colab status -s "$SESSION" ;;
  stop)
    colab stop -s "$SESSION" ;;
  r0)
    "$0" new; "$0" mount; "$0" prepare; "$0" autolabel; "$0" build r00; "$0" train r00 "${EPOCHS:-40}"; "$0" fetch r00
    echo "done: VM kept for inspection, release it with: $0 stop" ;;
  *)
    sed -n '2,14p' "$0"; exit 2 ;;
esac
