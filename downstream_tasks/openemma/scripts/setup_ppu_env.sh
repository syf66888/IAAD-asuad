#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv-ppu}"

python -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

export PIP_DISABLE_PIP_VERSION_CHECK=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
SITE_PACKAGES_DIR="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
cp "$ROOT_DIR/scripts/pip_sitecustomize.py" "$SITE_PACKAGES_DIR/sitecustomize.py"

if [[ -n "${ALIBABA_CLOUD_CREDENTIALS_URI:-}" ]]; then
  export AIEXT_REPO_AUTH_STS_TOKEN="$(curl -fsS "$ALIBABA_CLOUD_CREDENTIALS_URI")"
fi

pip install -r "$ROOT_DIR/requirements-ppu.txt"

echo "PPU-friendly OpenEMMA environment is ready."
echo "Activate with: source $VENV_DIR/bin/activate"
