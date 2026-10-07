#!/bin/bash
# Full correctness gate for SakuraCompiler: compiles every case in
# cases/functional, cases/h_functional and cases/performance2026 with the
# local compiler, then assembles / links / runs them on the RISC-V Ubuntu VM
# (QEMU) and diffs stdout+exit code against the reference .out files.
#
# Cases whose freshly-compiled assembly is byte-identical to the last verified
# run are skipped on the guest (see tools/test/qemu_verify.sh, build/.verify_cache).
#
# Requires the VM to be up with ssh on localhost:2222 (see tools/test/qemu_verify.sh).
#
#   ./tools/test/test.sh [suite ...]     default: functional h_functional performance2026
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== build =="
bash "$ROOT/tools/build/build.sh" || exit 1

suites=("$@")
[ ${#suites[@]} -eq 0 ] && suites=(functional h_functional performance2026)

./tools/test/qemu_verify.sh "${suites[@]}"
