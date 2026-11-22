#!/usr/bin/env bash
# Builds the HailoRT Python API (hailo_platform) for the Python that runs openNVE.
#
# Hailo publishes the Python API only for some Python versions (and Raspberry Pi OS only for its
# system Python), openNVE needs Python 3.14. This script compiles the Python bindings from the
# HailoRT source code of the HailoRT version that is installed on this computer, links them
# against the installed HailoRT library and installs the result as a wheel into that Python.
#
# Raspberry Pi 5 with AI Kit / AI HAT+ (Hailo-8L, Hailo-8), Raspberry Pi OS:
#
#     sudo apt install hailo-all                 # driver, firmware, HailoRT (libhailort)
#     sudo apt install git cmake g++ python3.14-dev   # or your own Python 3.14 with headers
#     source ~/opennve/.venv/bin/activate        # the virtual environment of openNVE
#     bash tools/build_hailort_python.sh
#     opennve system                             # HailoRT 4.x.x  1 Hailo device (.hef models)
#
# Options (environment variables):
#     PYTHON=/path/to/python      Python to build for (default: python in the active venv, else python3)
#     HAILORT_VERSION=4.23.0      HailoRT version (default: the installed one, from hailortcli / dpkg)
#     LIBHAILORT=/usr/lib/libhailort.so.4.23.0   the HailoRT library to link against
#     PYBIND11_VERSION=v3.0.1     pybind11 version (3.x is needed for Python 3.14)
#     WORK=~/.cache/opennve-hailort   build directory
#     JOBS=2                      parallel compile jobs (default: number of cores, 2 on 4 GB boards)
#
# Run it again after a HailoRT update (the Python API must have the same version as libhailort).
set -euo pipefail

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="${PYTHON:-$VIRTUAL_ENV/bin/python}"
fi
PYTHON="${PYTHON:-python3}"
PYTHON="$(command -v "$PYTHON")"
PYBIND11_VERSION="${PYBIND11_VERSION:-v3.0.1}"
WORK="${WORK:-$HOME/.cache/opennve-hailort}"

fail() { echo "error: $*" >&2; exit 1; }

for tool in git cmake g++; do
    command -v "$tool" >/dev/null || fail "$tool is missing (sudo apt install git cmake g++)"
done

PY_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
"$PYTHON" -c 'import sysconfig, os, sys; h = sysconfig.get_paths()["include"]; sys.exit(0 if os.path.exists(os.path.join(h, "Python.h")) else 1)' \
    || fail "Python.h of $PYTHON not found: install the development files (for instance python$PY_VERSION-dev)"

# installed HailoRT version
if [[ -z "${HAILORT_VERSION:-}" ]]; then
    if command -v hailortcli >/dev/null; then
        HAILORT_VERSION="$(hailortcli --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    fi
    if [[ -z "${HAILORT_VERSION:-}" ]] && command -v dpkg-query >/dev/null; then
        HAILORT_VERSION="$(dpkg-query -W -f='${Version}' hailort 2>/dev/null | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' || true)"
    fi
fi
[[ -n "${HAILORT_VERSION:-}" ]] || fail "HailoRT not found: install it first (Raspberry Pi OS: sudo apt install hailo-all), or set HAILORT_VERSION"

# the installed library
if [[ -z "${LIBHAILORT:-}" ]]; then
    LIBHAILORT="$( (ldconfig -p 2>/dev/null || true) | grep -oE "/[^ ]*libhailort\.so\.${HAILORT_VERSION//./\\.}" | head -1 || true)"
fi
if [[ -z "${LIBHAILORT:-}" ]]; then
    LIBHAILORT="$(find /usr/lib /usr/local/lib -name "libhailort.so.$HAILORT_VERSION" 2>/dev/null | head -1 || true)"
fi
[[ -n "${LIBHAILORT:-}" && -f "$LIBHAILORT" ]] || fail "libhailort.so.$HAILORT_VERSION not found, set LIBHAILORT=/path/to/libhailort.so.$HAILORT_VERSION"

echo "Python      $PYTHON ($PY_VERSION)"
echo "HailoRT     $HAILORT_VERSION ($LIBHAILORT)"
echo "pybind11    $PYBIND11_VERSION"
echo "build in    $WORK"

mkdir -p "$WORK"
SOURCE="$WORK/hailort-$HAILORT_VERSION"
if [[ ! -d "$SOURCE" ]]; then
    git clone --quiet --depth 1 --branch "v$HAILORT_VERSION" https://github.com/hailo-ai/hailort.git "$SOURCE"
fi
PYBIND11="$WORK/pybind11-$PYBIND11_VERSION"
if [[ ! -d "$PYBIND11" ]]; then
    git clone --quiet --depth 1 --branch "$PYBIND11_VERSION" https://github.com/pybind/pybind11.git "$PYBIND11"
fi

BINDINGS="$SOURCE/hailort/libhailort/bindings/python"
PLATFORM="$BINDINGS/platform"
BUILD="$WORK/build-$HAILORT_VERSION-py$PY_VERSION"
JOBS="${JOBS:-$(nproc)}"

cmake -S "$BINDINGS/src" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DPYBIND11_PYTHON_VERSION="$PY_VERSION" -DPYTHON_EXECUTABLE="$PYTHON" -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$PYTHON" \
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$PYBIND11" \
    -DLIBHAILORT_PATH="$LIBHAILORT" -DHAILORT_INCLUDE_DIR="$SOURCE/hailort/libhailort/include" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$BUILD/lib"
cmake --build "$BUILD" --target _pyhailort -j "$JOBS"

MODULE="$(find "$BUILD/lib" "$BUILD" -maxdepth 2 -name '_pyhailort*.so' | head -1)"
[[ -n "$MODULE" ]] || fail "_pyhailort was not built"
rm -f "$PLATFORM"/hailo_platform/pyhailort/_pyhailort*.so
cp "$MODULE" "$PLATFORM/hailo_platform/pyhailort/"

# Hailo's setup.py builds with its own options and pins numpy<2 (no wheels for Python 3.14);
# package the module that was just built instead. netifaces is only needed for Ethernet devices.
cat > "$WORK/setup_opennve.py" <<EOF
from setuptools import find_packages, setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="hailort",
    version="$HAILORT_VERSION",
    description="HailoRT Python API (hailo_platform), built for openNVE",
    url="https://github.com/hailo-ai/hailort",
    packages=find_packages(),
    package_data={"hailo_platform.pyhailort": ["_pyhailort*.so"]},
    install_requires=["numpy", "argcomplete", "contextlib2", "future", "netaddr"],
    entry_points={"console_scripts": ["hailo=hailo_platform.tools.hailocli.main:main"]},
    distclass=BinaryDistribution,
    zip_safe=False,
)
EOF
cp "$WORK/setup_opennve.py" "$PLATFORM/setup_opennve.py"
rm -rf "$PLATFORM/build" "$PLATFORM"/*.egg-info "$WORK/dist"
"$PYTHON" -m pip --version >/dev/null 2>&1 || "$PYTHON" -m ensurepip --upgrade >/dev/null
"$PYTHON" -m pip install --quiet --upgrade setuptools wheel
(cd "$PLATFORM" && "$PYTHON" setup_opennve.py --quiet bdist_wheel --dist-dir "$WORK/dist")
WHEEL="$(ls "$WORK"/dist/hailort-*.whl | head -1)"
"$PYTHON" -m pip install --force-reinstall "$WHEEL"

"$PYTHON" - <<'EOF'
import hailo_platform

print(f"hailo_platform {hailo_platform.__version__} installed")
try:
    devices = hailo_platform.Device.scan()
    print("Hailo devices:", ", ".join(devices) if devices else "none found")
except Exception as error:
    print("no Hailo device:", error)
EOF
echo "wheel: $WHEEL (install it in other environments with pip install)"
