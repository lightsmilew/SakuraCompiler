#!/bin/bash
# SakuraCompiler build / run helper.
#
#   ./tools/compile/run.sh -build                      incremental build
#   ./tools/compile/run.sh -rebuild                    clean + full build
#   ./tools/compile/run.sh -S file.sy [-o out.s]       compile one source to RISC-V asm
#   ./tools/compile/run.sh -qemu-test [suite...]       end-to-end check via QEMU VM
#                                                (suites: functional h_functional
#                                                 performance2026; default all three)
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BUILD_TREE="${BUILD_DIR:-build}"
case "$BUILD_TREE" in /*) ;; *) BUILD_TREE="$ROOT/$BUILD_TREE" ;; esac
COMP="${COMP:-$BUILD_TREE/compiler}"

case "${1:-}" in
  -build)
    bash "$ROOT/tools/build/build.sh"
    ;;
  -rebuild)
    bash "$ROOT/tools/build/build.sh" clean
    ;;
  -S)
    shift
    "$COMP" "$@"
    ;;
  -qemu-test)
    shift
    suites=("$@")
    [ ${#suites[@]} -eq 0 ] && suites=(functional h_functional performance2026)
    ./tools/test/qemu_verify.sh "${suites[@]}"
    ;;
  *)
    echo "usage: $0 {-build|-rebuild|-S file.sy [-o out.s]|-qemu-test [suite...]}"
    exit 1
    ;;
esac
