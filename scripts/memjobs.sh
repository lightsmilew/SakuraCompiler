#!/bin/bash
# memjobs.sh -- print a memory-aware parallel-job count for this machine.
#
#   memjobs            -> a job count (positive integer, >= 1)
#   memjobs -v         -> also print how it was derived on stderr
#
# The count is the smaller of (ncpus, floor(available-RAM / 2 GiB) capped
# at 8).  Using all of nproc for a parallel C++ build on a memory-poor VM
# (default WSL2 is often ~8 GiB) can exhaust RAM and take the whole VM down
# with it, so the default parallelism stays conservative.
#
# Environment: MEMJOBS_MAX (default 8), MEMJOBS_PER_GB (default 2 GiB per
# job).  Honour an explicit JOBS from the caller instead of this default.
set -u

memjobs() {
  local ncpu
  ncpu=$(nproc 2>/dev/null || echo 4)
  local max=${MEMJOBS_MAX:-8}
  local per_gb=${MEMJOBS_PER_GB:-2}

  # Available memory in MiB from /proc/meminfo (Linux/WSL).  Fall back to
  # nproc when the file is unavailable or unreadable.
  local avail_mib=""
  if [ -r /proc/meminfo ]; then
    avail_mib=$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo 2>/dev/null)
  fi
  if [ -z "$avail_mib" ] || [ "$avail_mib" -le 0 ]; then
    echo "$ncpu"
    [ "${1:-}" = "-v" ] && echo "memjobs: no MemAvailable, using ncpu=$ncpu" >&2
    return 0
  fi

  # Memory-based parallelism: one job per per_gb GiB of available RAM.
  local mjobs
  mjobs=$((avail_mib / (per_gb * 1024)))
  [ "$mjobs" -lt 1 ] && mjobs=1

  # Never exceed the CPU count or the caller's ceiling.
  local jobs=$mjobs
  [ "$jobs" -gt "$ncpu" ] && jobs=$ncpu
  [ "$jobs" -gt "$max" ] && jobs=$max

  if [ "${1:-}" = "-v" ]; then
    echo "memjobs: ncpu=$ncpu mem_avail=${avail_mib}MiB -> jobs=$jobs" >&2
  fi
  echo "$jobs"
}

# Called as a command: print the value (optionally -v for the explanation).
if [ "$(basename "$0" 2>/dev/null)" = "memjobs.sh" ] ||
   [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
  memjobs "${1:-}"
fi
