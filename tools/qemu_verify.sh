#!/bin/bash
# End-to-end correctness check: compiles the cases with the local compiler,
# pushes the .s/.in/.out files to the RISC-V Ubuntu VM (QEMU, ssh :2222) and
# runs tools/guest_test.sh there (assemble + link with libsysy_riscv + run +
# diff against .out).
#
# Verification cache: the assembly verified on a previous run is kept under
# $CACHE_DIR.  When the freshly-produced .s of a case is byte-identical to the
# last verified one, and its sources / expected I/O / compiler are unchanged
# (the cache key), the case is reported as CACHED and is NOT re-run on the
# guest.  Only cases whose assembly changed (or that have never been verified)
# are transferred and executed under QEMU.  This makes repeat runs of the full
# regression nearly free after the first clean pass.
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
STAGING_RUN="${STAGING_RUN:-/tmp/saku_verify_stage_run}"
CACHE_DIR="${CACHE_DIR:-$ROOT/build/.verify_cache}"
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
# corrupt each other.  Clean up any stragglers first, and wipe the remote tree
# so a subset re-transfer can never mix stale files from an earlier run.  The
# [.] avoids pkill matching our own command line.
ssh_cmd "pkill -9 -f 'guest_test[.]sh' 2>/dev/null; pkill -9 -f '[.]elf' 2>/dev/null; sleep 1; rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT' && echo guest_ok" || { echo "ssh failed"; exit 1; }

# Cache key: compiler binary + source + expected I/O.  Any of these changing
# invalidates the previous result even if the assembly text did not change.
comp_hash=$(sha256sum "$COMP" 2>/dev/null | cut -d' ' -f1) || { echo "ERROR: cannot read $COMP"; exit 1; }
key_for() {
  # $1 suite  $2 base -> prints the verification key
  local suite="$1" base="$2" k f
  k="$comp_hash|$(sha256sum "$ROOT/cases/$suite/$base.sy" | cut -d' ' -f1)"
  for x in in out; do
    f="$ROOT/cases/$suite/$base.$x"
    [ -f "$f" ] && k="$k|$(sha256sum "$f" | cut -d' ' -f1)"
  done
  printf '%s' "$k"
}

echo "== compiling locally =="
rm -rf "$STAGING" "$STAGING_RUN"
mkdir -p "$STAGING" "$STAGING_RUN"
FAILED_COMPILE=0
NEEDS_RUN=0
CACHED=0
declare -A CACHED_SUITE
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
      # a previously-passing cached result is no longer trustworthy
      rm -f "$CACHE_DIR/$suite/$base.ok"
      continue
    fi
    if [ ! -f "$src/$base.out" ]; then
      # nothing to diff against on the guest; not run, not cached
      echo "NO_OUT $suite/$base"
      continue
    fi
    if [ -f "$CACHE_DIR/$suite/$base.ok" ] \
       && [ "$(cat "$CACHE_DIR/$suite/$base.ok" 2>/dev/null)" = "$(key_for "$suite" "$base")" ] \
       && [ -f "$CACHE_DIR/$suite/$base.s" ] \
       && cmp -s "$CACHE_DIR/$suite/$base.s" "$STAGING/$suite/$base.s"; then
      echo "CACHED $suite/$base"
      CACHED=$((CACHED + 1))
      CACHED_SUITE[$suite]=$(( ${CACHED_SUITE[$suite]:-0} + 1 ))
      continue
    fi
    NEEDS_RUN=$((NEEDS_RUN + 1))
    mkdir -p "$STAGING_RUN/$suite"
    cp "$STAGING/$suite/$base.s" "$STAGING_RUN/$suite/"
    [ -f "$src/$base.in" ]  && cp -f "$src/$base.in"  "$STAGING_RUN/$suite/"
    [ -f "$src/$base.out" ] && cp -f "$src/$base.out" "$STAGING_RUN/$suite/"
  done
done

run_suites=()
for suite in "${SUITES[@]}"; do
  [ -d "$STAGING_RUN/$suite" ] && run_suites+=("$suite")
done

guest_log=/tmp/saku_guest_run.log
if [ ${#run_suites[@]} -gt 0 ]; then
  echo "== transfer to guest =="
  for suite in "${run_suites[@]}"; do
    scp_cmd -r "$STAGING_RUN/$suite" "${QEMU_USER}@${QEMU_HOST}:$REMOTE_ROOT/"
  done
  scp_cmd "$ROOT/tools/guest_test.sh" "${QEMU_USER}@${QEMU_HOST}:/tmp/guest_test.sh"

  echo "== guest run =="
  args=""
  for suite in "${run_suites[@]}"; do args="$args $REMOTE_ROOT/$suite"; done
  ssh_cmd "chmod +x /tmp/guest_test.sh && GUEST_VERIFY_MODE=1 /tmp/guest_test.sh$args" | tee "$guest_log"
  guest_rc=${PIPESTATUS[0]}
else
  echo "== nothing to re-verify on the guest (all cached) =="
  guest_rc=0
fi

# Refresh the cache from the per-case guest verdicts.  Only cases that just
# PASSed get their assembly + key recorded; a FAIL/ASSEMBLE/LINK/TIMEOUT drops
# any stale ok marker so the case is retried next time.
run_pass=0
run_fail=0
cur_suite=""
if [ ${#run_suites[@]} -gt 0 ]; then
  while IFS= read -r line; do
    case "$line" in
      "===== suite: "*)
        cur_suite=${line#*"===== suite: "}; cur_suite=${cur_suite%" ====="}
        ;;
      "PASS "*)
        base=${line#*"PASS "}
        mkdir -p "$CACHE_DIR/$cur_suite"
        cp "$STAGING_RUN/$cur_suite/$base.s" "$CACHE_DIR/$cur_suite/$base.s"
        printf '%s' "$(key_for "$cur_suite" "$base")" > "$CACHE_DIR/$cur_suite/$base.ok"
        run_pass=$((run_pass + 1))
        ;;
      "FAIL "*|"ASSEMBLE_FAIL "*|"LINK_FAIL "*|"TIMEOUT "*)
        base=${line#*" "}
        rm -f "$CACHE_DIR/$cur_suite/$base.ok"
        run_fail=$((run_fail + 1))
        ;;
    esac
  done < "$guest_log"

  if [ $((run_pass + run_fail)) -ne "$NEEDS_RUN" ]; then
    echo "ERROR: guest verdicts ($((run_pass + run_fail))) do not match cases sent ($NEEDS_RUN); run aborted?"
    exit 1
  fi
fi

echo "===== PASS=$((run_pass + CACHED)) FAIL=$run_fail ====="
for suite in "${SUITES[@]}"; do
  n=${CACHED_SUITE[$suite]:-0}
  [ "$n" -gt 0 ] && echo "cached $suite: $n"
done
echo "compile failures: $FAILED_COMPILE"
exit $((run_fail + FAILED_COMPILE))
