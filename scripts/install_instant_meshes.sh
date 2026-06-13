#!/usr/bin/env bash
# Build Instant Meshes from source (BSD-3-Clause) for aarch64/x86_64 Linux.
# Usage: ./scripts/install_instant_meshes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IM_DIR="${ROOT}/thirdparty/instant-meshes"
BUILD_DIR="${IM_DIR}/build"

echo "[install_instant_meshes] root=${ROOT}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "[install_instant_meshes] cmake not found; install cmake or activate venv." >&2
  exit 1
fi

if [[ ! -d "${IM_DIR}/.git" ]]; then
  echo "[install_instant_meshes] cloning wjakob/instant-meshes ..."
  git clone --recursive --depth 1 https://github.com/wjakob/instant-meshes.git "${IM_DIR}"
else
  echo "[install_instant_meshes] updating ${IM_DIR} ..."
  git -C "${IM_DIR}" pull --ff-only || true
  git -C "${IM_DIR}" submodule update --init --recursive
fi

# Required on Ubuntu/Debian for GLFW/X11 (build will fail without these headers).
MISSING_PKGS=()
for pkg in libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxxf86vm-dev build-essential; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    MISSING_PKGS+=("$pkg")
  fi
done
if ((${#MISSING_PKGS[@]} > 0)); then
  echo "[install_instant_meshes] Install system packages first:" >&2
  echo "  sudo apt-get install -y ${MISSING_PKGS[*]}" >&2
  exit 1
fi

# CMake 3.30+ rejects CMP0042 OLD in bundled GLFW (nanogui submodule).
GLFW_CMAKE="${IM_DIR}/ext/nanogui/ext/glfw/CMakeLists.txt"
if [[ -f "${GLFW_CMAKE}" ]] && grep -q 'CMP0042 OLD' "${GLFW_CMAKE}"; then
  echo "[install_instant_meshes] patching GLFW for CMake 4.x (CMP0042 NEW) ..."
  sed -i 's/cmake_policy(SET CMP0042 OLD)/cmake_policy(SET CMP0042 NEW)/' "${GLFW_CMAKE}"
fi

ITT_CONFIG="${IM_DIR}/ext/tbb/src/tbb/tools_api/ittnotify_config.h"
if [[ -f "${ITT_CONFIG}" ]] && ! grep -q 'defined __aarch64__' "${ITT_CONFIG}"; then
  echo "[install_instant_meshes] patching bundled TBB for aarch64 (ittnotify_config.h) ..."
  python3 - <<'PY' "${ITT_CONFIG}"
import pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text()
text = text.replace(
    "#  elif defined _M_ARM || __arm__\n#    define ITT_ARCH ITT_ARCH_ARM",
    "#  elif defined _M_ARM || defined __arm__\n#    define ITT_ARCH ITT_ARCH_ARM\n"
    "#  elif defined __aarch64__\n#    define ITT_ARCH ITT_ARCH_ARM",
)
if "#else\n#define __TBB_machine_fetchadd4" not in text:
    text = text.replace(
        "#elif ITT_ARCH==ITT_ARCH_ARM || ITT_ARCH==ITT_ARCH_PPC64\n"
        "#define __TBB_machine_fetchadd4(addr, val) __sync_fetch_and_add(addr, val)\n"
        "#endif /* ITT_ARCH==ITT_ARCH_IA64 */",
        "#elif ITT_ARCH==ITT_ARCH_ARM || ITT_ARCH==ITT_ARCH_PPC64\n"
        "#define __TBB_machine_fetchadd4(addr, val) __sync_fetch_and_add(addr, val)\n"
        "#else\n"
        "#define __TBB_machine_fetchadd4(addr, val) __sync_fetch_and_add(addr, val)\n"
        "#endif /* ITT_ARCH==ITT_ARCH_IA64 */",
    )
p.write_text(text)
PY
fi

cmake -S "${IM_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "${BUILD_DIR}" -j "$(nproc)"

BIN=""
for candidate in "${BUILD_DIR}/Instant Meshes" "${BUILD_DIR}/instant-meshes"; do
  if [[ -x "${candidate}" ]]; then
    BIN="${candidate}"
    break
  fi
done
if [[ -z "${BIN}" ]]; then
  echo "[install_instant_meshes] binary missing under ${BUILD_DIR}" >&2
  exit 1
fi

echo "[install_instant_meshes] OK: ${BIN}"
echo "export INSTANT_MESHES_BIN='${BIN}'"
