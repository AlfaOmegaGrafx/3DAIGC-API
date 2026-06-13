#!/usr/bin/env bash
# Build traveller59/spconv (+ FindDefinition/cumm) for ./venv for TRELLIS.
# spconv is installed editable from ~/.cache/3daigc-spconv/spconv-<tag>/ so JIT
# can create spconv/core_cc (non-editable wheel installs omit it and break imports).
#
# PyTorch cu128 is built for CUDA 12.x; torch.utils.cpp_extension refuses to compile
# CUDA extensions when the *major* nvcc version differs (e.g. default nvcc 13.x).
# Point CUDA_HOME at a CUDA 12.x toolkit (NVIDIA /usr/local/cuda-12.x or distro
# nvidia-cuda-toolkit, often /usr/lib/nvidia-cuda-toolkit).
#
# Prereqs (typical Ubuntu):
#   sudo apt install python3.12-dev build-essential ninja-build cmake git
#   # cumm/spconv: use GCC 12 host + C++17 (script sets this when g++-12 is installed):
#   sudo apt install gcc-12 g++-12
#
# Optional: SPARSE_BACKEND=torchsparse if you prefer torchsparse over spconv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/venv"
if [[ ! -x "$VENV/bin/pip" ]]; then
  echo "Missing venv at $VENV (create with: python3 -m venv venv && ./venv/bin/pip install -U pip wheel)" >&2
  exit 1
fi

detect_cuda12_home() {
  local d cand ver maj
  for d in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-12; do
    if [[ -x "$d/bin/nvcc" ]]; then
      ver="$("$d/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
      maj="${ver%%.*}"
      if [[ "$maj" == "12" ]]; then
        echo "$d"
        return 0
      fi
    fi
  done
  cand="/usr/lib/nvidia-cuda-toolkit"
  if [[ -x "$cand/bin/nvcc" ]]; then
    ver="$("$cand/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
    maj="${ver%%.*}"
    if [[ "$maj" == "12" ]]; then
      echo "$cand"
      return 0
    fi
  fi
  return 1
}

if [[ -z "${CUDA_HOME:-}" ]] || [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if detected="$(detect_cuda12_home)"; then
    CUDA_HOME="$detected"
    echo "[build_spconv_local] Auto-selected CUDA_HOME=$CUDA_HOME (CUDA 12.x nvcc)." >&2
  fi
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
NVCC="$CUDA_HOME/bin/nvcc"
if [[ ! -x "$NVCC" ]]; then
  echo "nvcc not found at $NVCC. Install CUDA 12.x (e.g. cuda-toolkit-12-8) or nvidia-cuda-toolkit, then:" >&2
  echo "  export CUDA_HOME=/usr/local/cuda-12.8   # or your 12.x prefix" >&2
  echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >&2
  echo "  $0" >&2
  exit 1
fi

rel="$( "$NVCC" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' )"
maj="${rel%%.*}"
if [[ "$maj" != "12" ]]; then
  echo "CUDA toolkit at CUDA_HOME=$CUDA_HOME is version $rel (major $maj)." >&2
  echo "PyTorch cu128 needs CUDA 12.x nvcc. Install CUDA 12.8 (or 12.x), set CUDA_HOME, put \$CUDA_HOME/bin on PATH, then:" >&2
  echo "  $0" >&2
  exit 1
fi

pyver="$("$VENV/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
find_python_headers() {
  local candidates=(
    "/usr/include/python${pyver}"
    "$HOME/.local/py${pyver//./}-dev/usr/include/python${pyver}"
    "$HOME/.local/python${pyver}-dev/usr/include/python${pyver}"
    "${PYTHON_INCLUDE_DIR:-}"
  )
  for d in "${candidates[@]}"; do
    [[ -n "$d" && -f "$d/Python.h" ]] && { echo "$d"; return 0; }
  done
  return 1
}
if pyinc="$(find_python_headers)"; then
  # Debian/Ubuntu: Python.h lives under .../include/python3.12/ but pyconfig.h pulls in
  # <aarch64-linux-gnu/python3.12/pyconfig.h> from .../include/aarch64-linux-gnu/python3.12/
  # (libpython*-dev). Both parent .../include and the python subdir must be on CPATH.
  pyinc_root="$(dirname "$pyinc")"
  export CPATH="$pyinc_root:$pyinc${CPATH:+:$CPATH}"
  echo "[build_spconv_local] Using Python headers: $pyinc (CPATH includes $pyinc_root)" >&2
else
  echo "Python development headers not found (Python.h)." >&2
  echo "Either:" >&2
  echo "  - sudo apt install python${pyver}-dev   (system-wide), or" >&2
  echo "  - apt download python${pyver}-dev libpython${pyver}-dev && dpkg -x ./*.deb $HOME/.local/py${pyver//./}-dev/" >&2
  exit 1
fi

# nvcc < 12.4 on glibc 2.39+ aarch64 trips on SVE intrinsic types in <bits/math-vector.h>.
# Detect and warn before we burn a long build.
if [[ "$(uname -m)" == "aarch64" ]]; then
  glibc_ver="$(ldd --version 2>/dev/null | head -n1 | sed -n 's/.* \([0-9]\+\.[0-9]\+\)$/\1/p')"
  case "$rel" in
    12.0|12.1|12.2|12.3)
      cat >&2 <<EOF
[build_spconv_local] WARNING: nvcc $rel on aarch64 with glibc ${glibc_ver:-?} commonly fails to compile
  due to undefined __Float32x4_t / __SVFloat32_t types in glibc's <bits/math-vector.h>.
  Install CUDA 12.4+ (recommended 12.8 to match PyTorch cu128) and re-run.
  Workarounds (not guaranteed): -D__ARM_NEON_SVE_BRIDGE, include <arm_neon.h> first.
EOF
      ;;
  esac
fi

# cumm (spconv dep): device/host helpers in half.h need C++17; nvcc defaults to C++14.
# Host g++-13 + libstdc++ often cascades parse errors in .cu units; prefer g++-12 when present.
if command -v g++-12 >/dev/null 2>&1 && [[ -z "${CXX:-}" ]]; then
  export CC="${CC:-gcc-12}"
  export CXX="${CXX:-g++-12}"
  echo "[build_spconv_local] Using CC=$CC CXX=$CXX for cumm/spconv (recommended on Ubuntu 24.04 + CUDA 12.x)." >&2
fi

export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
# nvcc: host compiler + C++17 (matches cumm; avoids half_t / copysign parse failures with g++-13).
if [[ -n "${CXX:-}" ]]; then
  _ccbin="$(command -v "$CXX")"
  if [[ -z "${NVCC_PREPEND_FLAGS:-}" ]]; then
    export NVCC_PREPEND_FLAGS="-ccbin=$_ccbin --std=c++17 -Xcompiler=-std=c++17"
  fi
elif command -v g++-12 >/dev/null 2>&1 && [[ -z "${NVCC_PREPEND_FLAGS:-}" ]]; then
  export NVCC_PREPEND_FLAGS="-ccbin=$(command -v g++-12) --std=c++17 -Xcompiler=-std=c++17"
fi
# CUDA 12.8+ warns once per .cu when -gencode includes sm_<75; harmless but very noisy.
if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]] && [[ "${NVCC_PREPEND_FLAGS}" != *"deprecated-gpu-targets"* ]]; then
  export NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS -Wno-deprecated-gpu-targets"
fi
if [[ -n "${CXX:-}" ]]; then
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v "$CXX")}"
fi

"$VENV/bin/pip" install -q "ninja>=1.11" "cmake>=3.20" pccm packaging

# pccm/ccimport default std="c++14"; cumm v0.7.13 headers need C++17 (inline __device__
# functions at namespace scope, etc.). spconv's build.py calls build_pybind without `std=`,
# so we patch the installed defaults in-venv. Idempotent.
"$VENV/bin/python" - <<'PY'
from pathlib import Path
import re, site, sys
roots = list(map(Path, site.getsitepackages())) + [Path(p) for p in sys.path if p]
targets = [
    "pccm/builder/pybind.py",
    "pccm/extension.py",
    "ccimport/core.py",
    "ccimport/extension.py",
]
# Matches: std="c++14" / std='c++14' / std: Optional[str] = "c++14" / std=DEFAULT, etc.
pat = re.compile(r"""(\bstd\b[^=\n]*=\s*)(['"])c\+\+14\2""")
seen, patched = set(), []
for r in roots:
    for t in targets:
        p = r / t
        if not p.is_file() or p in seen:
            continue
        seen.add(p)
        src = p.read_text()
        new, n = pat.subn(lambda m: f"{m.group(1)}{m.group(2)}c++17{m.group(2)}", src)
        if n and new != src:
            p.write_text(new)
            patched.append(f"{p} ({n} replacement{'s' if n != 1 else ''})")
if patched:
    print("[build_spconv_local] Patched std=c++17 in:")
    for line in patched:
        print("  -", line)
else:
    print("[build_spconv_local] pccm/ccimport already at c++17 (no patch needed).")
PY

# cumm fallback gencode list includes sm_52, which triggers __CUDA_ARCH__ < 530 legacy
# branch in tensorview/gemm/dtypes/half.h that fails to parse on nvcc 12.x. Restrict to
# sm_75+ by default: nvcc 12.8 warns (and will drop) offline targets below sm_75; sm_70
# was in our old default. cumm v0.7.13 caps supported_arches at 9.0; +PTX JIT for GB10+.
# For Volta (sm_70) set: export CUMM_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
if [[ -z "${CUMM_CUDA_ARCH_LIST:-}" ]]; then
  export CUMM_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
  echo "[build_spconv_local] Defaulting CUMM_CUDA_ARCH_LIST=$CUMM_CUDA_ARCH_LIST (no sm_<75; avoids nvcc 12.8 deprecation spam + half.h sm_52 path)." >&2
fi
# If CUMM_CUDA_ARCH_LIST was unset during an earlier import, cumm uses nvidia-smi; an
# unrecognized GPU name falls back to arches including 5.2. CUDA 12.8 aarch64 nvcc then
# errors: Unsupported gpu architecture 'compute_52'. Gencode is baked into
# spconv/build/core_cc/build.ninja — scrub stale trees so this script's export wins.

CUMM_TAG="${CUMM_TAG:-v0.7.13}"
CUMM_DIR="${CUMM_SRC:-${XDG_CACHE_HOME:-$HOME/.cache}/3daigc-spconv/cumm-${CUMM_TAG}}"
mkdir -p "$(dirname "$CUMM_DIR")"
if [[ ! -d "$CUMM_DIR/.git" ]]; then
  rm -rf "$CUMM_DIR"
  git clone --depth 1 --branch "$CUMM_TAG" --recursive https://github.com/FindDefinition/cumm.git "$CUMM_DIR"
else
  git -C "$CUMM_DIR" fetch --depth 1 origin "refs/tags/${CUMM_TAG}:refs/tags/${CUMM_TAG}" 2>/dev/null || true
  git -C "$CUMM_DIR" checkout -q "$CUMM_TAG"
  git -C "$CUMM_DIR" submodule update --init --recursive
fi
(
  cd "$CUMM_DIR"
  "$VENV/bin/pip" install --no-build-isolation -q -e .
)

# cumm/common.py:_get_cuda_include_lib() only recognizes conda-style targets/x86_64-linux
# layouts; on aarch64 + system CUDA 12.8 (targets/sbsa-linux/, plus include/+lib64/), it
# falls through to a hardcoded "/usr/local/cuda" which on this host is a symlink to CUDA
# 13.0 via /etc/alternatives. Inject a $CUDA_HOME-first override at the top of the
# function. Idempotent.
"$VENV/bin/python" - "$CUMM_DIR/cumm/common.py" "$CUDA_HOME" <<'PY'
import sys, re
from pathlib import Path
src_path = Path(sys.argv[1])
cuda_home = sys.argv[2]
src = src_path.read_text()
marker = "# [3DAIGC-API] prefer $CUDA_HOME"
if marker in src:
    print(f"[build_spconv_local] cumm/common.py already patched (CUDA_HOME-first).")
    sys.exit(0)
sig = "def _get_cuda_include_lib():"
body_open = "if _CACHED_CUDA_INCLUDE_LIB is None:"
pat = re.compile(
    r"(def _get_cuda_include_lib\(\):\s*\n\s*global _CACHED_CUDA_INCLUDE_LIB\s*\n\s*if _CACHED_CUDA_INCLUDE_LIB is None:\s*\n)"
)
inject = (
    "        " + marker + ": handle aarch64 + system CUDA toolkit layouts.\n"
    "        import os as _os\n"
    "        _ch = _os.environ.get('CUDA_HOME') or _os.environ.get('CUDA_PATH')\n"
    "        if _ch:\n"
    "            _chp = Path(_ch)\n"
    "            for _inc_sub, _lib_sub in (\n"
    "                ('include', 'lib64'),\n"
    "                ('targets/sbsa-linux/include', 'targets/sbsa-linux/lib'),\n"
    "                ('targets/aarch64-linux/include', 'targets/aarch64-linux/lib'),\n"
    "                ('targets/x86_64-linux/include', 'targets/x86_64-linux/lib'),\n"
    "            ):\n"
    "                _inc = _chp / _inc_sub\n"
    "                _lib = _chp / _lib_sub\n"
    "                if (_inc / 'cuda.h').exists() and ((_lib / 'libcudart.so').exists() or (_lib / 'stubs' / 'libcudart.so').exists()):\n"
    "                    _CACHED_CUDA_INCLUDE_LIB = ([_inc], _lib)\n"
    "                    return _CACHED_CUDA_INCLUDE_LIB\n"
)
new, n = pat.subn(lambda m: m.group(1) + inject, src, count=1)
if n != 1:
    sys.stderr.write("[build_spconv_local] could not locate _get_cuda_include_lib() to patch.\n")
    sys.exit(1)
src_path.write_text(new)
print(f"[build_spconv_local] Patched cumm/common.py to prefer $CUDA_HOME ({cuda_home}).")
PY

# spconv/build.py only runs pccm build_pybind (creates spconv/core_cc) when
# project_is_installed && project_is_editable. A plain `pip install .` copies
# Python sources to site-packages, so editable is false and core_cc never exists
# (ModuleNotFoundError: spconv.core_cc). Use a persistent clone + editable install
# so PACKAGE_ROOT/../.. has .gitignore/setup.py (pccm.utils.project_is_editable).
# Override checkout: SPCONV_SRC=/path/to/spconv  and/or  SPCONV_TAG=v2.3.8
SPCONV_TAG="${SPCONV_TAG:-v2.3.8}"
SPCONV_DIR="${SPCONV_SRC:-${XDG_CACHE_HOME:-$HOME/.cache}/3daigc-spconv/spconv-${SPCONV_TAG}}"
mkdir -p "$(dirname "$SPCONV_DIR")"
if [[ ! -d "$SPCONV_DIR/.git" ]]; then
  rm -rf "$SPCONV_DIR"
  git clone --depth 1 --branch "$SPCONV_TAG" --recursive https://github.com/traveller59/spconv.git "$SPCONV_DIR"
else
  git -C "$SPCONV_DIR" fetch --depth 1 origin "refs/tags/${SPCONV_TAG}:refs/tags/${SPCONV_TAG}" 2>/dev/null || true
  git -C "$SPCONV_DIR" checkout -q "$SPCONV_TAG" 2>/dev/null || git -C "$SPCONV_DIR" checkout -q "tags/$SPCONV_TAG"
  git -C "$SPCONV_DIR" submodule update --init --recursive
fi
if [[ "$(uname -m)" == "aarch64" && -f "$SPCONV_DIR/build/core_cc/build.ninja" ]] \
  && grep -Fq "arch=compute_52" "$SPCONV_DIR/build/core_cc/build.ninja" 2>/dev/null; then
  echo "[build_spconv_local] Removing stale $SPCONV_DIR/build/core_cc (compute_52 gencode; aarch64 CUDA 12.x nvcc rejects it — will regenerate with CUMM_CUDA_ARCH_LIST)." >&2
  rm -rf "$SPCONV_DIR/build/core_cc"
fi
(
  cd "$SPCONV_DIR"
  # traveller59/cumm is gone from GitHub; build cumm from FindDefinition first, then drop
  # cumm from build-system requires (PEP 517 cannot see the editable install otherwise).
  "$VENV/bin/python" - <<'PY'
from pathlib import Path
import re
p = Path("pyproject.toml")
text = p.read_text()
if '"cumm>=' not in text and "'cumm>=" not in text:
    print("[build_spconv_local] pyproject.toml: cumm already removed from build-system (ok).")
else:
    text2 = re.sub(r'"cumm>=[^"]+",?\s*', '', text)
    text2 = re.sub(r',\s*"cumm>=[^"]+"', '', text2)
    if text2 == text:
        raise SystemExit("failed to patch cumm out of pyproject.toml [build-system].requires")
    p.write_text(text2)
    print("[build_spconv_local] Patched cumm out of pyproject.toml [build-system].requires.")
PY
  echo "[build_spconv_local] pip install -e $SPCONV_DIR (first import will JIT-build spconv/core_cc)." >&2
  "$VENV/bin/pip" uninstall -y spconv >/dev/null 2>&1 || true
  "$VENV/bin/pip" install --no-build-isolation -q -v -e .
)

echo "spconv editable-linked into $VENV from $SPCONV_DIR. Quick check (may compile core_cc):"
cd "$ROOT"
"$VENV/bin/python" -c "import spconv.pytorch as s; print('spconv OK', s.__file__)"
