#!/usr/bin/env bash
# Install system Blender for UniRig on Linux aarch64 (DGX Spark).
# UniRig uses bpy via utils/blender_runtime.py subprocess when pip bpy is unavailable.
set -euo pipefail

echo "[install_blender_unirig] Installing Blender (Ubuntu/Debian arm64) ..."
sudo apt-get update -qq
sudo apt-get install -y blender

BLENDER="$(command -v blender)"
echo "[install_blender_unirig] blender: $("$BLENDER" --version | head -1)"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BLENDER_BIN="$BLENDER"
export DAIGC_ROOT="$ROOT"

echo ""
echo "Add to your shell or scripts/env_local_gpu.sh:"
echo "  export BLENDER_BIN=$BLENDER"
echo ""
echo "Verify from API venv:"
echo "  cd $ROOT && source scripts/env_local_gpu.sh"
echo "  python -c \"from utils.blender_runtime import find_blender_binary, bpy_importable; print('bpy in venv:', bpy_importable()); print('blender:', find_blender_binary())\""
