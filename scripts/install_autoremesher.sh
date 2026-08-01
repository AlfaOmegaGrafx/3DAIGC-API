#!/usr/bin/env bash
# Build AutoRemesher from source (MIT) for aarch64/x86_64 Linux.
# Usage: ./scripts/install_autoremesher.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AR_DIR="${ROOT}/thirdparty/autoremesher"

echo "[install_autoremesher] root=${ROOT}"

# AutoRemesher QT += core widgets opengl (no multimedia/svg required).
# Ubuntu package names: qt5-qmake / qtbase5-dev / libtbb-dev / libgl1-mesa-dev.
# (libqt5multimedia5-dev does not exist on Noble — use qtmultimedia5-dev only if needed.)
if ! command -v qmake >/dev/null 2>&1; then
  echo "[install_autoremesher] qmake not found; install Qt5 build tools first:" >&2
  echo "  sudo apt-get install -y build-essential qt5-qmake qtbase5-dev qttools5-dev-tools libtbb-dev libgl1-mesa-dev" >&2
  exit 1
fi

MISSING_PKGS=()
for pkg in build-essential qt5-qmake qtbase5-dev qttools5-dev-tools libtbb-dev libgl1-mesa-dev; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    MISSING_PKGS+=("$pkg")
  fi
done
if ((${#MISSING_PKGS[@]} > 0)); then
  echo "[install_autoremesher] Install system packages first:" >&2
  echo "  sudo apt-get install -y ${MISSING_PKGS[*]}" >&2
  exit 1
fi

if [[ ! -d "${AR_DIR}/.git" ]]; then
  echo "[install_autoremesher] cloning huxingyi/autoremesher ..."
  git clone --depth 1 https://github.com/huxingyi/autoremesher.git "${AR_DIR}"
else
  echo "[install_autoremesher] updating ${AR_DIR} ..."
  git -C "${AR_DIR}" pull --ff-only || true
fi

# Upstream hardcodes -march=x86-64-v2 / -flto, and Geogram assumes x86 atomics on
# all non-Android Linux. Both break on aarch64 (DGX Spark).
PRO="${AR_DIR}/autoremesher.pro"
ARCH="$(uname -m)"
if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
  echo "[install_autoremesher] patching for ${ARCH} (march/LTO + Geogram ARM atomics) ..."

  if [[ -f "${PRO}" ]]; then
    sed -i \
      -e 's/-march=x86-64-v2/-march=native/g' \
      -e 's/QMAKE_CXXFLAGS_RELEASE += -flto -funroll-loops -march=native/QMAKE_CXXFLAGS_RELEASE += -funroll-loops -march=native/g' \
      -e 's/QMAKE_LFLAGS_RELEASE += -flto/QMAKE_LFLAGS_RELEASE -= -flto/g' \
      "${PRO}"
    if grep -q -- '-flto' "${PRO}"; then
      sed -i -e 's/ -flto//g' -e 's/-flto //g' "${PRO}"
    fi
  fi

  ATOMICS="${AR_DIR}/thirdparty/geogram/geogram-1.8.3/src/lib/geogram/basic/atomics.h"
  if [[ -f "${ATOMICS}" ]] && ! grep -q 'defined(__aarch64__)' "${ATOMICS}"; then
    # Prefer GCC builtins (Android path) over x86 pause/bts on aarch64.
    python3 - <<'PY' "${ATOMICS}"
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text()
old = """#  elif defined(GEO_OS_ANDROID)
#    define GEO_USE_ANDROID_ATOMICS
#  else
#    define GEO_USE_X86_ATOMICS
#  endif"""
new = """#  elif defined(GEO_OS_ANDROID) || defined(__aarch64__) || defined(__arm64__)
#    define GEO_USE_ANDROID_ATOMICS
#  else
#    define GEO_USE_X86_ATOMICS
#  endif"""
if old not in text:
    raise SystemExit(f"atomics.h: expected Linux atomics block not found in {p}")
text = text.replace(old, new, 1)
needle = "#elif defined(GEO_USE_ANDROID_ATOMICS)\n\n/** A mutex for Android */"
repl = (
    "#elif defined(GEO_USE_ANDROID_ATOMICS)\n\n"
    "inline void geo_pause() {\n"
    "}\n\n"
    "/** A mutex for Android */"
)
if needle not in text:
    raise SystemExit(f"atomics.h: ANDROID atomics block not found in {p}")
text = text.replace(needle, repl, 1)
p.write_text(text)
print(f"patched {p}")
PY
  fi

  COMMON="${AR_DIR}/thirdparty/geogram/geogram-1.8.3/src/lib/geogram/basic/common.h"
  if [[ -f "${COMMON}" ]] && grep -q 'defined(__x86_64)$' "${COMMON}"; then
    # aarch64 was incorrectly classified as GEO_ARCH_32.
    sed -i 's/#if defined(__x86_64)$/#if defined(__x86_64) || defined(__aarch64__) || defined(__arm64__)/' "${COMMON}"
  fi

  THREAD_SYNC="${AR_DIR}/thirdparty/geogram/geogram-1.8.3/src/lib/geogram/basic/thread_sync.h"
  if [[ -f "${THREAD_SYNC}" ]] && ! grep -q 'GEOGRAM_AARCH64_DEFAULT_SPINLOCK' "${THREAD_SYNC}"; then
    # Bit-packed SpinLockArray uses x86 bts; on ARM use the portable vector path.
    # Do NOT fold aarch64 into the GEO_OS_APPLE block (that pulls AvailabilityMacros.h).
    python3 - <<'PY' "${THREAD_SYNC}"
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text()
marker = "GEOGRAM_AARCH64_DEFAULT_SPINLOCK"
if marker in text:
    print(f"already patched {p}")
    raise SystemExit(0)
anchor = "#ifdef GEO_OS_APPLE\n# define GEO_USE_DEFAULT_SPINLOCK_ARRAY\n"
if anchor not in text:
    raise SystemExit(f"thread_sync.h: APPLE default spinlock block not found in {p}")
insert = (
    "/* GEOGRAM_AARCH64_DEFAULT_SPINLOCK */\n"
    "#if defined(__aarch64__) || defined(__arm64__)\n"
    "# ifndef GEO_USE_DEFAULT_SPINLOCK_ARRAY\n"
    "#  define GEO_USE_DEFAULT_SPINLOCK_ARRAY\n"
    "# endif\n"
    "#endif\n\n"
    + anchor
)
text = text.replace(anchor, insert, 1)
old2 = "#elif defined(GEO_OS_LINUX) \n\n        /**\n         * \\brief An array of light-weight synchronisation"
new2 = "#elif defined(GEO_OS_LINUX) && defined(GEO_USE_X86_ATOMICS)\n\n        /**\n         * \\brief An array of light-weight synchronisation"
if old2 not in text:
    raise SystemExit(f"thread_sync.h: LINUX SpinLockArray block not found in {p}")
text = text.replace(old2, new2, 1)
p.write_text(text)
print(f"patched {p}")
PY
  fi
fi

pushd "${AR_DIR}" >/dev/null
# Drop stale objects/Makefiles after .pro / arch changes (LTO objs are especially sticky).
rm -f Makefile Makefile.Debug Makefile.Release .qmake.stash
rm -rf obj moc
qmake
make -j "$(nproc)"
popd >/dev/null

BIN="${AR_DIR}/autoremesher"
if [[ ! -x "${BIN}" ]]; then
  # Some qmake configs put the binary next to Makefile under release/
  for candidate in "${AR_DIR}/autoremesher" "${AR_DIR}/release/autoremesher"; do
    if [[ -x "${candidate}" ]]; then
      BIN="${candidate}"
      break
    fi
  done
fi
if [[ ! -x "${BIN}" ]]; then
  echo "[install_autoremesher] binary missing under ${AR_DIR}" >&2
  exit 1
fi

echo "[install_autoremesher] OK: ${BIN}"
echo "export AUTOREMESHER_BIN='${BIN}'"
