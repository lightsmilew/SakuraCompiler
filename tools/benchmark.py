#!/usr/bin/env python3
"""Build comparable rv64gc binaries and benchmark on an already running VM.

Example (WSL/Linux):
  QEMU_PASS=... python3 tools/benchmark.py --variant before=build/compiler.before \
    --variant after=build/compiler --llvm clang-18 --library build/libsysy_riscv.a

Use --compile-only for assembly/RA diagnostics without a guest. Compiler work
is sequential by default: large SysY global initializers can use gigabytes.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=COMPILER")
    parser.add_argument("--llvm", help="Clang executable, e.g. clang-18")
    parser.add_argument("--suite", type=pathlib.Path, default=ROOT / "cases/performance2026")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "build/performance")
    parser.add_argument("--library", type=pathlib.Path)
    parser.add_argument("--cc", default="riscv64-linux-gnu-gcc")
    parser.add_argument("--guest", default="ubuntu@127.0.0.1")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--remote-dir", default="/home/ubuntu/saku-performance")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore assembly/link caches")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    variants = {}
    for spec in args.variant:
        name, sep, binary = spec.partition("=")
        if not sep or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            parser.error("variant must be NAME=COMPILER with a simple directory name")
        if name in variants or (name == "llvm" and args.llvm):
            parser.error("duplicate variant name")
        variants[name] = pathlib.Path(binary).resolve()
    if args.llvm:
        variants["llvm"] = pathlib.Path(shutil.which(args.llvm) or args.llvm).resolve()
    if not variants:
        parser.error("supply --variant and/or --llvm")
    if not args.compile_only and (not args.library or not args.library.is_file()):
        parser.error("--library must point to libsysy_riscv.a for executable tests")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    manifest = {"target": "rv64gc/lp64d", "runs": args.runs,
                "suite": str(args.suite.resolve()), "variants": {}, "cases": {}}
    pass_env = {k: v for k, v in os.environ.items() if k.startswith("SAKU_")}
    manifest["pass_environment"] = pass_env
    llvm_flags = ["--target=riscv64-linux-gnu", "-march=rv64gc", "-mabi=lp64d",
                  "-O3", "-fno-builtin", "-fno-slp-vectorize", "-fno-addrsig",
                  "-ffp-contract=off"]
    link_inputs = None
    if not args.compile_only:
        cc_path = pathlib.Path(shutil.which(args.cc) or args.cc).resolve()
        link_inputs = {"cc_sha256": sha256(cc_path), "runtime_sha256": sha256(args.library)}
        for library in ("libc.a", "libgcc.a"):
            path = pathlib.Path(subprocess.check_output(
                [args.cc, "-print-file-name=" + library], text=True).strip())
            if path.is_file():
                link_inputs[library] = sha256(path)
        manifest["link_inputs"] = link_inputs
    sources = sorted(args.suite.glob("*.sy"))
    if args.cases:
        sources = [s for s in sources if s.stem in args.cases]
    if not sources:
        parser.error("no matching cases")
    for src in sources:
        manifest["cases"][src.stem] = {"source_sha256": sha256(src)}
        for ext in (".in", ".out"):
            file = src.with_suffix(ext)
            if file.exists():
                if ext == ".out":
                    normalized = b" ".join(file.read_bytes().split())
                    (out / "cases" / (file.name + ".sha256")).write_text(
                        hashlib.sha256(normalized).hexdigest())
                else:
                    shutil.copy2(file, out / "cases" / file.name)
                manifest["cases"][src.stem][ext[1:] + "_sha256"] = sha256(file)
    for name, binary in variants.items():
        dest = out / name
        dest.mkdir(exist_ok=True)
        manifest["variants"][name] = {"compiler": str(binary), "sha256": sha256(binary)}
        flags = llvm_flags if name == "llvm" else ["-O2", "--pass-stats"]
        manifest["variants"][name]["flags"] = flags
        if name == "llvm":
            manifest["variants"][name]["version"] = subprocess.check_output(
                [str(binary), "--version"], text=True).splitlines()[0]
        for src in sources:
            asm = dest / (src.stem + ".s")
            if name == "llvm":
                cpp = dest / (src.stem + ".cpp")
                cpp.write_text((ROOT / "scripts/sysy_prelude.h").read_text() + src.read_text())
                command = [str(binary), *flags, "-S", str(cpp), "-o", str(asm)]
            else:
                command = [str(binary), str(src.resolve()), "-O2", "--pass-stats", "-o", str(asm)]
            signature = dict(compiler=manifest["variants"][name]["sha256"],
                             source=sha256(src), flags=flags, environment=pass_env)
            if name == "llvm":
                signature["prelude"] = sha256(ROOT / "scripts/sysy_prelude.h")
            key = json.dumps(signature, sort_keys=True)
            keyfile = dest / (src.stem + ".compile.key")
            cached = (not args.force and asm.is_file() and keyfile.is_file()
                      and keyfile.read_text() == key)
            if not cached:
                with (dest / (src.stem + ".compile.log")).open("w") as log:
                    subprocess.run(command, stdout=log, stderr=log, check=True, timeout=180)
                keyfile.write_text(key)
            if not args.compile_only:
                link_key = json.dumps(dict(assembly=sha256(asm), **link_inputs), sort_keys=True)
                linkfile = dest / (src.stem + ".link.key")
                exe = asm.with_suffix(".elf")
                if args.force or not exe.is_file() or not linkfile.is_file() or linkfile.read_text() != link_key:
                    subprocess.run([args.cc, "-march=rv64gc", "-mabi=lp64d", "-static",
                                    str(asm), str(args.library.resolve()), "-o", str(exe)], check=True)
                    linkfile.write_text(link_key)
            print("CACHED" if cached else "BUILT", name, src.stem, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.compile_only:
        return 0
    shutil.copy2(ROOT / "tools/benchmark_guest.py", out / "benchmark_guest.py")
    archive = out / "payload.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for file in [out / "benchmark_guest.py", out / "manifest.json"]:
            tar.add(file, arcname=file.name)
        tar.add(out / "cases", arcname="cases")
        for name in variants:
            for file in (out / name).glob("*.elf"):
                tar.add(file, arcname=name + "/" + file.name)
    env = dict(os.environ)
    prefix = []
    if env.get("QEMU_PASS"):
        env["SSHPASS"] = env["QEMU_PASS"]
        prefix = ["sshpass", "-e"]
    options = ["-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new"]
    def ssh(command):
        return subprocess.run([*prefix, "ssh", *options, "-p", str(args.port), args.guest, command], env=env)
    def scp(src, dst):
        subprocess.run([*prefix, "scp", *options, "-P", str(args.port), src, dst], env=env, check=True)
    remote = args.remote_dir.rstrip("/")
    if not remote.startswith("/") or any(c not in "/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in remote):
        parser.error("remote-dir must be a simple absolute path")
    if ssh("mkdir -p " + shlex.quote(remote)).returncode:
        raise RuntimeError("VM SSH unavailable; start the VM once before benchmarking")
    scp(str(archive), args.guest + ":" + remote + "/payload.tar.gz")
    scp(str(out / "benchmark_guest.py"), args.guest + ":" + remote + "/benchmark_guest.py")
    command = ("cd " + shlex.quote(remote) + " && python3 benchmark_guest.py . --unpack payload.tar.gz && python3 benchmark_guest.py . "
               + " ".join(map(shlex.quote, variants)) + " --runs " + str(args.runs)
               + " --timeout " + str(args.timeout))
    # Always select this invocation's inputs, even when `out` contains an
    # earlier run of a larger suite.
    command += " --cases " + " ".join(shlex.quote(src.stem) for src in sources)
    result = ssh(command)
    for name in ("measurements.jsonl", "summary.json"):
        scp(args.guest + ":" + remote + "/" + name, str(out / name))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
