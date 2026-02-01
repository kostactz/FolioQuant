#!/usr/bin/env bash
set -euo pipefail

# Run helper for FolioQuant project
# Usage:
#   ./run.sh install    -> create virtualenv and install dependencies
#   ./run.sh            -> run Streamlit app (prefers venv), demo, or tests as fallback
#   ./run.sh test       -> run the test suite via pytest (prefers venv python)

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="$PWD/venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Helper function to check Python version
check_python_version() {
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Error: Python ($PYTHON) not found. Please install Python 3.10 or later."
    exit 1
  fi
  
  local py_version=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
  if [ -z "$py_version" ]; then
    echo "Error: Could not determine Python version"
    exit 1
  fi
  
  local major=$(echo "$py_version" | cut -d. -f1)
  local minor=$(echo "$py_version" | cut -d. -f2)
  
  if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 10 ]); then
    echo "Error: Python 3.10+ required (found $py_version)"
    exit 1
  fi
  
  echo "Found Python $py_version"
}

# Helper function to ensure venv module is available
ensure_venv_module() {
  if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
    echo "Python venv module not available. Attempting to install..."
    
    # Detect the distribution and install python3-venv
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing python3-venv via apt-get..."
      if ! sudo apt-get update >/dev/null 2>&1; then
        echo "Warning: apt-get update failed, attempting install anyway..."
      fi
      if ! sudo apt-get install -y --no-install-recommends python3-venv >/dev/null 2>&1; then
        echo "Error: Failed to install python3-venv via apt-get"
        exit 1
      fi
    elif command -v yum >/dev/null 2>&1; then
      echo "Installing python3-venv via yum..."
      if ! sudo yum install -y python3-venv >/dev/null 2>&1; then
        echo "Error: Failed to install python3-venv via yum"
        exit 1
      fi
    elif command -v pacman >/dev/null 2>&1; then
      echo "Installing python3-venv via pacman..."
      if ! sudo pacman -S --noconfirm python >/dev/null 2>&1; then
        echo "Error: Failed to install python3-venv via pacman"
        exit 1
      fi
    elif command -v brew >/dev/null 2>&1; then
      echo "Note: macOS detected. If venv is missing, reinstall Python via: brew install python3"
      exit 1
    else
      echo "Error: Could not determine package manager. Please manually install python3-venv."
      exit 1
    fi
    
    # Verify the module is now available
    if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
      echo "Error: venv module still not available after installation"
      exit 1
    fi
  fi
}

case "${1:-}" in
  install)
    echo "=== FolioQuant Installation ==="
    check_python_version
    ensure_venv_module
    
    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment..."
      "$PYTHON" -m venv "$VENV_DIR"
    else
      echo "Virtual environment already exists at $VENV_DIR"
    fi
    
    # Ensure pip is available inside the venv. Some systems create venvs
    # without a bundled pip module; try ensurepip and fall back to get-pip.py.
    echo "Verifying and installing pip..."
    if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
      if "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1; then
        echo "Bootstrapped pip in venv via ensurepip"
      else
        echo "ensurepip not available; attempting to download get-pip.py"
        TMP_GET_PIP="/tmp/get-pip.py"
        if command -v curl >/dev/null 2>&1; then
          curl -sS https://bootstrap.pypa.io/get-pip.py -o "$TMP_GET_PIP"
        else
          python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py','/tmp/get-pip.py')
PY
        fi
        "$VENV_PY" "$TMP_GET_PIP"
        rm -f "$TMP_GET_PIP"
      fi
    fi

    # Upgrade packaging tools
    echo "Upgrading pip, setuptools, and wheel..."
    "$VENV_PY" -m pip install --upgrade pip setuptools wheel >/dev/null

    if [ -f requirements.txt ]; then
      echo "Installing project dependencies from requirements.txt..."
      "$VENV_PIP" install -r requirements.txt
    else
      echo "requirements.txt not found in project root"
      exit 1
    fi
    echo ""
    echo "=== Setup Complete ==="
    echo "Activate the virtual environment with:"
    echo "  source venv/bin/activate"
    echo ""
    exit 0
    ;;

  test)
    if [ -x "$VENV_PY" ]; then
      "$VENV_PY" -m pytest -q
    else
      python -m pytest -q
    fi
    exit $?
    ;;
  run)
    # explicit run -> fallthrough to default behavior
    ;;
  "")
    ;;
  *)
    # allow passthrough to underlying commands if needed
    ;;
esac

# Run Dash app (prefers venv)
if [ -x "$VENV_PY" ]; then
  export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
  # Run Python unbuffered to see logs immediately
  export PYTHONUNBUFFERED=1
  "$VENV_PY" -m src.app.dash_app --product BTC-USD --ofi-window 100 --port 8501
  exit $?
fi

# Fallback to system python with Dash
if command -v python3 >/dev/null 2>&1; then
  export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
  # Run Python unbuffered to see logs immediately
  export PYTHONUNBUFFERED=1
  python3 -m src.app.dash_app --product BTC-USD --ofi-window 100 --port 8501
  exit $?
fi

echo "No Dash installation or demo found. Running tests as fallback."
if [ -x "$VENV_PY" ]; then
  "$VENV_PY" -m pytest -q
else
  python -m pytest -q
fi
