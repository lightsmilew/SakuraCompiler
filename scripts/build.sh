#!/bin/bash
# build.sh -- build the SakuraCompiler driver binary.
#
# Usage:
#   ./scripts/build.sh            incremental build (default)
#   ./scripts/build.sh clean      wipe build/ and rebuild from scratch
#
# Environment:
#   BUILD_DIR   build tree location            (default: <repo>/build)
#   JOBS        parallel jobs  (default: memory-aware, see scripts/memjobs.sh)
#
# The only required toolchain is CMake >= 3.10 and a C++17 compiler; the
# ANTLR4 runtime is vendored under 3rd_party/, so nothing is fetched from the
# network.  The resulting driver is <BUILD_DIR>/compiler.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
if [ -n "${JOBS:-}" ]; then
  JOBS="$JOBS"
else
  # shellcheck source=/dev/null
  . "$ROOT/scripts/memjobs.sh"
  JOBS="$(memjobs)"
fi

case "${1:-}" in
  clean|-clean|--clean)
    rm -rf "$BUILD_DIR"
    ;;
  -h|--help|help)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    ;;
esac

echo "== configure: cmake -S . -B $BUILD_DIR =="
cmake -S . -B "$BUILD_DIR" || exit 1
echo "== build: cmake --build $BUILD_DIR -j$JOBS =="
cmake --build "$BUILD_DIR" -j"$JOBS" || exit 1
echo "== done: $BUILD_DIR/compiler =="
