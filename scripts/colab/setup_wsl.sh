#!/usr/bin/env bash
# One-time setup of google-colab-cli inside WSL Ubuntu (the CLI supports Linux / macOS only).
#   bash scripts/colab/setup_wsl.sh
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv (Python package manager) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# keyrings.alt: WSL has no unlocked OS keyring, so the OAuth token goes to a file readable by this user only
uv tool install --upgrade google-colab-cli --with keyrings.alt

line='export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring'
grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring

colab version
echo
echo "Next: open a NEW WSL terminal (or: source ~/.bashrc), then"
echo "  cd /mnt/d/AiThucChien/other/hackathon-l2-team56"
echo "  scripts/colab/pipeline.sh new      # first time: paste the Google authorization code"
