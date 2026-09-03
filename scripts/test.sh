#!/bin/bash
# Full correctness gate for SakuraCompiler: compiles every case in
# cases/functional, cases/h_functional and cases/performance2026 with the
# local compiler, then assembles / links / runs them on the RISC-V Ubuntu VM
# (QEMU) and diffs stdout+exit code against the reference .out files.
#
# Requires the VM to be up with ssh on localhost:2222 (see tools/qemu_verify.sh).
#
#   ./scripts/test.sh [suite ...]     default: functional h_functional performance2026
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== build =="
cmake -S . -B build >/dev/null
cmake --build build -j"$(nproc 2>/dev/null || echo 4)" || exit 1

suites=("$@")
[ ${#suites[@]} -eq 0 ] && suites=(functional h_functional performance2026)

./tools/qemu_verify.sh "${suites[@]}"
