#!/bin/bash
# Runs on the RISC-V Ubuntu VM (QEMU).  For every .s file in the given suite
# directories (under $WORK), assembles it, links it against libsysy_riscv.a,
# runs it (feeding <name>.in when present) and compares stdout + exit code
# against <name>.out (whitespace-normalised, timer lines filtered).  Each pass
# also prints "PASS <name>" so the host can update its verification cache.
#
# Usage: guest_test.sh <dir1> [dir2 ...]
# Directories are absolute paths on the guest.
set -uo pipefail

CROSS=riscv64-linux-gnu-
MARCH="${MARCH:-rv64gc}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
LIB=$(ls "${HOME}/libsysy_riscv.a" "${HOME}/riscv/libsysy_riscv.a" 2>/dev/null | head -1)
[ -n "$LIB" ] || { echo "ERROR: libsysy_riscv.a not found"; exit 2; }

normalize() { tr -d '\r' < "$1" | tr -s '[:space:]' ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

PASS=0
FAIL=0
LOG=/tmp/guest_test_failures.log
: > "$LOG"

for dir in "$@"; do
  [ -d "$dir" ] || continue
  echo "===== suite: $(basename "$dir") ====="
  for f in "$dir"/*.s; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .s)
    exp="$dir/$base.out"
    [ -f "$exp" ] || continue

    if ! ${CROSS}as -march="$MARCH" "$f" -o "/tmp/${base}.o"; then
      echo "ASSEMBLE_FAIL $base"
      FAIL=$((FAIL + 1))
      continue
    fi
    if ! ${CROSS}gcc -march="$MARCH" -static "/tmp/${base}.o" -L"$(dirname "$LIB")" -lsysy_riscv \
         -o "/tmp/${base}.elf"; then
      echo "LINK_FAIL $base"
      FAIL=$((FAIL + 1))
      continue
    fi

    inp="$dir/$base.in"
    if [ -f "$inp" ]; then
      timeout "${TIMEOUT_SECONDS}s" "/tmp/${base}.elf" < "$inp" > "/tmp/${base}.out.act" 2>&1
      rc=$?
    else
      timeout "${TIMEOUT_SECONDS}s" "/tmp/${base}.elf" > "/tmp/${base}.out.act" 2>&1
      rc=$?
    fi
    if [ $rc -eq 124 ]; then
      echo "TIMEOUT $base"
      FAIL=$((FAIL + 1))
      continue
    fi
    # exit code appended directly (the expected .out embeds it, e.g. putint(-5)
    # then rc=0 yields "-50").
    printf '%d' "$rc" >> "/tmp/${base}.out.act"

    # strip timing instrumentation the library emits at exit.  Its text can be
    # glued to the final output token and even split a number across the line
    # break, so remove everything from the marker up to the trailing "us" (the
    # timer id part may look negative, hence the loose .*?) anywhere it occurs.
    perl -0pe '
      s/Timer@.*?us\n?//g;
      s/TOTAL:.*?us\n?//g;' "/tmp/${base}.out.act" \
      > "/tmp/${base}.out.filtered"

    normalize "$exp" > "/tmp/${base}.exp.norm"
    normalize "/tmp/${base}.out.filtered" > "/tmp/${base}.act.norm"
    if diff -q "/tmp/${base}.exp.norm" "/tmp/${base}.act.norm" > /dev/null; then
      PASS=$((PASS + 1))
      # per-case verdict so the host can update its verification cache
      echo "PASS $base"
    else
      echo "FAIL $base"
      FAIL=$((FAIL + 1))
      echo "--- $base ---" >> "$LOG"
      diff -u "/tmp/${base}.exp.norm" "/tmp/${base}.act.norm" >> "$LOG"
    fi
    rm -f "/tmp/${base}.o" "/tmp/${base}.elf"
  done
done

# when driven by qemu_verify.sh the host prints the canonical grand total
# (it also counts CACHED cases); standalone runs get the local summary here
if [ "${GUEST_VERIFY_MODE:-0}" -ne 1 ]; then
  echo "===== PASS=$PASS FAIL=$FAIL ====="
fi
if [ "$FAIL" -gt 0 ]; then echo "see $LOG"; fi
exit "$FAIL"
