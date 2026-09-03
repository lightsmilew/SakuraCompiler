#!/bin/bash
# SakuraCompiler build / run helper.
#
#   ./scripts/run.sh -build                      incremental build
#   ./scripts/run.sh -rebuild                    clean + full build
#   ./scripts/run.sh -S file.sy [-o out.s]       compile one source to RISC-V asm
#   ./scripts/run.sh -qemu-test [suite...]       end-to-end check via QEMU VM
#                                                (suites: functional h_functional
#                                                 performance2026; default all three)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="build"
JOBS="$(nproc 2>/dev/null || echo 4)"
COMP="$BUILD_DIR/compiler"

case "${1:-}" in
  -build)
    cmake -S . -B "$BUILD_DIR" >/dev/null
    cmake --build "$BUILD_DIR" -j"$JOBS"
    ;;
  -rebuild)
    rm -rf "$BUILD_DIR"
    cmake -S . -B "$BUILD_DIR" >/dev/null
    cmake --build "$BUILD_DIR" -j"$JOBS"
    ;;
  -S)
    shift
    "$COMP" "$@"
    ;;
  -qemu-test)
    shift
    suites=("$@")
    [ ${#suites[@]} -eq 0 ] && suites=(functional h_functional performance2026)
    ./tools/qemu_verify.sh "${suites[@]}"
    ;;
  *)
    echo "usage: $0 {-build|-rebuild|-S file.sy [-o out.s]|-qemu-test [suite...]}"
    exit 1
    ;;
esac
