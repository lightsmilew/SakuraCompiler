#!/bin/bash
# End-to-end correctness check: compiles the cases with the local compiler,
# pushes the .s/.in/.out files to the RISC-V Ubuntu VM (QEMU, ssh :2222) and
# runs tools/guest_test.sh there (assemble + link with libsysy_riscv + run +
# diff against .out).
#
# Usage:
#   ./tools/qemu_verify.sh                      # cases/functional
#   ./tools/qemu_verify.sh functional h_functional
#   ./tools/qemu_verify.sh performance2026
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QEMU_HOST="${QEMU_HOST:-localhost}"
QEMU_PORT="${QEMU_PORT:-2222}"
QEMU_USER="${QEMU_USER:-ubuntu}"
QEMU_PASS="${QEMU_PASS:-WSJ040511}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/saku_verify}"
COMP="${COMP:-$ROOT/build/compiler}"
STAGING="${STAGING:-/tmp/saku_verify_stage}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8"

SUITES=("$@")
[ ${#SUITES[@]} -gt 0 ] || SUITES=(functional)

ssh_cmd() { sshpass -p "$QEMU_PASS" ssh $SSH_OPTS -p "$QEMU_PORT" "${QEMU_USER}@${QEMU_HOST}" "$@"; }
scp_cmd()  { sshpass -p "$QEMU_PASS" scp $SSH_OPTS -P "$QEMU_PORT" "$@"; }

echo "== check VM ssh =="
if ! timeout 5 bash -c "echo >/dev/tcp/${QEMU_HOST}/${QEMU_PORT}" 2>/dev/null; then
  echo "ERROR: QEMU ssh port ${QEMU_PORT} closed - start the VM first."
  exit 1
fi
# a previous run killed on the host can leave guest_test.sh alive on the guest
# (no tty means no SIGHUP), and two concurrent runs on the same /tmp files
# corrupt each other.  Clean up any stragglers first.  The [.] avoids pkill
# matching our own command line.
ssh_cmd "pkill -9 -f 'guest_test[.]sh' 2>/dev/null; pkill -9 -f '[.]elf' 2>/dev/null; sleep 1; mkdir -p '$REMOTE_ROOT' && echo guest_ok" || { echo "ssh failed"; exit 1; }

echo "== compiling locally =="
rm -rf "$STAGING"
mkdir -p "$STAGING"
FAILED_COMPILE=0
for suite in "${SUITES[@]}"; do
  src="$ROOT/cases/$suite"
  [ -d "$src" ] || { echo "skip missing suite $suite"; continue; }
  mkdir -p "$STAGING/$suite"
  for f in "$src"/*.sy; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .sy)
    if ! timeout 120s "$COMP" "$f" -S -o "$STAGING/$suite/$base.s" >/tmp/saku_compile.log 2>&1; then
      echo "COMPILE_FAIL $suite/$base"
      tail -3 /tmp/saku_compile.log
      FAILED_COMPILE=$((FAILED_COMPILE + 1))
      continue
    fi
    [ -f "$src/$base.in" ]  && cp -f "$src/$base.in"  "$STAGING/$suite/"
    [ -f "$src/$base.out" ] && cp -f "$src/$base.out" "$STAGING/$suite/"
  done
done

echo "== transfer to guest =="
for suite in "${SUITES[@]}"; do
  [ -d "$STAGING/$suite" ] || continue
  scp_cmd -r "$STAGING/$suite" "${QEMU_USER}@${QEMU_HOST}:$REMOTE_ROOT/"
done
scp_cmd "$ROOT/tools/guest_test.sh" "${QEMU_USER}@${QEMU_HOST}:/tmp/guest_test.sh"

echo "== guest run =="
args=""
for suite in "${SUITES[@]}"; do args="$args $REMOTE_ROOT/$suite"; done
ssh_cmd "chmod +x /tmp/guest_test.sh && /tmp/guest_test.sh$args"
guest_rc=$?

echo "compile failures: $FAILED_COMPILE"
exit $((guest_rc + FAILED_COMPILE))
