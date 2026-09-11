#!/bin/bash
# Compile a SysY case with the system clang as a *reference* for RISC-V codegen.
#   scripts/llvm_ref.sh <case.sy> [out.s] [extra clang flags...]
# SysY is a C subset, so we prepend the runtime declarations and tell clang to
# treat the file as C++ (file-scope `const int N` is a constant expression in
# C++, which is how the array bounds in these cases are meant to be read).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
src=$1; shift
out=${1:-/tmp/llvm_ref.s}; shift 2>/dev/null || true
CLANG=${CLANG:-clang-18}
tmp=$(mktemp /tmp/llvmref_XXXXXX.cpp)
cat "$ROOT/scripts/sysy_prelude.h" "$src" > "$tmp"
"$CLANG" --target=riscv64-unknown-linux-gnu -march=rv64gc -mabi=lp64d \
  -O3 -fno-builtin -fno-slp-vectorize -fno-addrsig -S -o "$out" -x c++ "$tmp" "$@"
rc=$?
rm -f "$tmp"
exit $rc
