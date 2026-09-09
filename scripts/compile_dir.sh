#!/bin/bash
# compile_dir.sh -- batch-compile every SysY source under one or more
# directories with the SakuraCompiler driver, writing one artifact per file.
#
# Usage:
#   ./scripts/compile_dir.sh DIR... -o OUTDIR [MODE] [-- compiler flags]
#
# Modes (exactly one; default --asm):
#   --asm       RISC-V64 assembly        -> OUTDIR/<dir>/<base>.s
#   --affine    affine-layer IR dump     -> OUTDIR/<dir>/<base>.affine.mlir
#   --scf       scf-layer IR dump        -> OUTDIR/<dir>/<base>.scf.mlir
#   --cf        cf-layer IR dump         -> OUTDIR/<dir>/<base>.cf.mlir
#   --final     final pre-ISel IR dump   -> OUTDIR/<dir>/<base>.final.mlir
#   --ir        affine+scf+cf in one run -> OUTDIR/<dir>/<base>.ir.txt
#
# Files are written under OUTDIR/<label>/ where <label> is the basename of
# each input directory (e.g. cases/functional -> OUTDIR/functional/), so
# running the tool over several suites never mixes files.
#
# Compiler flags may be passed after "--" (or after the mode), e.g. -O0 -O1
# -O2 (default -O2).  Everything after "--" is forwarded verbatim to the
# compiler.  IR-dump modes discard the generated assembly (the driver always
# runs the back-end; stdout is redirected to the .mlir artifact).
#
# Environment:
#   COMP        compiler binary          (default: <repo>/build/compiler)
#   JOBS        parallel jobs            (default: nproc, capped at 16)
#   KEEP_GOING  set to 1 to continue past a failing file (default: stop)
#
# Examples:
#   ./scripts/compile_dir.sh cases/functional -o /tmp/out          # assembly
#   ./scripts/compile_dir.sh cases/tensor cases/h_functional \
#       -o /tmp/out --cf -O1                                        # cf IR
#   ./scripts/compile_dir.sh cases/functional -o /tmp/out --asm -O0 -- --dump-cf
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMP="${COMP:-$ROOT/build/compiler}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
[ "$JOBS" -gt 16 ] && JOBS=16
KEEP_GOING="${KEEP_GOING:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"

if [ ! -x "$COMP" ]; then
  echo "ERROR: compiler not found at $COMP - run ./scripts/build.sh first" >&2
  exit 2
fi

MODE="asm"
OUTDIR=""
DIRS=()
FLAGS=()
i=0
while [ $i -lt $# ]; do
  a="${*:$((i+1)):1}"
  case "$a" in
    --asm|--affine|--scf|--cf|--final|--ir)
      MODE="${a#--}"
      i=$((i+1))
      ;;
    -o)
      OUTDIR="${*:$((i+2)):1}"
      [ $# -gt $((i+1)) ] || OUTDIR=""
      i=$((i+2))
      ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    --)
      # everything after -- is a compiler flag (verbatim)
      j=$((i+1))
      while [ $j -le $# ]; do FLAGS+=("${*:$j:1}"); j=$((j+1)); done
      i=$#
      ;;
    -*) FLAGS+=("$a"); i=$((i+1)) ;;
    *) DIRS+=("$a"); i=$((i+1)) ;;
  esac
done

if [ -z "$OUTDIR" ] || [ ${#DIRS[@]} -eq 0 ]; then
  echo "usage: $0 DIR... -o OUTDIR [--asm|--affine|--scf|--cf|--final|--ir] [-O0|-O1|-O2 ...]"
  echo "       (compiler flags after the mode are forwarded; see header of $0)"
  exit 1
fi

mkdir -p "$OUTDIR"

# Per-mode: driver flag and output extension.
case "$MODE" in
  asm)    DFLAG=();            EXT=s ;;
  affine) DFLAG=(--dump-affine); EXT=affine.mlir ;;
  scf)    DFLAG=(--dump-scf);   EXT=scf.mlir ;;
  cf)     DFLAG=(--dump-cf);    EXT=cf.mlir ;;
  final)  DFLAG=(--dump-final); EXT=final.mlir ;;
  ir)     DFLAG=(--dump-ir);    EXT=ir.txt ;;
esac

SRCS=()
for d in "${DIRS[@]}"; do
  if [ -f "$d" ]; then
    SRCS+=("$d")
  elif [ -d "$d" ]; then
    for f in "$d"/*.sy; do [ -f "$f" ] && SRCS+=("$f"); done
  else
    echo "WARN: skip (not a file/dir): $d"
  fi
done
if [ ${#SRCS[@]} -eq 0 ]; then
  echo "ERROR: no .sy sources found under: ${DIRS[*]}" >&2
  exit 1
fi
echo "== compiling ${#SRCS[@]} files (mode=$MODE) -> $OUTDIR =="

label_of() {
  local d="$1"
  d="${d%/}"
  echo "${d##*/}"
}

fail=0
fail_txt="$OUTDIR/.failures.txt"
: > "$fail_txt"

run_one() {
  local src="$1" d="$2" label ext
  label="$(label_of "$d")"
  case "$MODE" in
    asm) ext=s ;;
    affine) ext=affine.mlir ;;
    scf) ext=scf.mlir ;;
    cf) ext=cf.mlir ;;
    final) ext=final.mlir ;;
    ir) ext=ir.txt ;;
  esac
  local base
  base="$(basename "$src" .sy)"
  local outdir="$OUTDIR/$label"
  mkdir -p "$outdir"
  local out="$outdir/$base.$ext"
  local log="$outdir/$base.compile.log"

  if [ "$MODE" = "asm" ]; then
    # assembly -> -o writes the artifact; diagnostics to a per-file log.
    timeout "$TIMEOUT_SECONDS"s "$COMP" "$src" -S -o "$out" "${FLAGS[@]}" \
      >"$log" 2>&1
  else
    # IR dump goes to stdout; send the always-generated assembly to /dev/null.
    timeout "$TIMEOUT_SECONDS"s "$COMP" "$src" "${DFLAG[@]}" -o /dev/null \
      "${FLAGS[@]}" >"$out" 2>"$log"
  fi
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL $label/$(basename "$src") (rc=$rc)"
    { echo "== $src (rc=$rc) =="; tail -5 "$log"; } >> "$fail_txt"
    return 1
  fi
  echo "OK   $label/$(basename "$src") -> $out"
  rm -f "$log"
  return 0
}

pids=()
slots=0
failed_some=0
run_batch() {
  local src="$1" d="$2"
  if [ "$KEEP_GOING" -ne 1 ] && [ $failed_some -ne 0 ]; then
    return
  fi
  run_one "$src" "$d" &
  pids+=($!)
  slots=$((slots + 1))
  if [ $slots -ge "$JOBS" ]; then
    wait "${pids[@]}"
    for p in "${pids[@]}"; do
      wait "$p" || failed_some=1
    done
    pids=()
    slots=0
  fi
}

for d in "${DIRS[@]}"; do
  if [ -f "$d" ]; then
    run_batch "$d" "$(dirname "$d")"
  else
    for f in "$d"/*.sy; do
      [ -f "$f" ] || continue
      run_batch "$f" "$d"
    done
  fi
  if [ "$KEEP_GOING" -ne 1 ] && [ $failed_some -ne 0 ]; then
    break
  fi
done
wait "${pids[@]}" 2>/dev/null
for p in "${pids[@]}"; do wait "$p" 2>/dev/null || failed_some=1; done

if [ $failed_some -ne 0 ]; then
  echo "==== failures (see $fail_txt): ===="
  cat "$fail_txt"
  exit 1
fi
echo "==== all ${#SRCS[@]} files compiled OK (mode=$MODE) ===="
