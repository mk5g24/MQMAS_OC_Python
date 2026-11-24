#!/usr/bin/env python3
"""Create a Python environment and build the MQMAS C++ core.

This script is intentionally standard-library-only so it can be run with a
fresh Python installation.  It detects the current operating system, the
architecture of the Python interpreter (which must match the ctypes library),
and a suitable CMake generator/compiler.

Typical usage from the repository root:

    python bootstrap.py

Useful variants:

    python bootstrap.py --clean --recreate-venv
    python bootstrap.py --skip-tests
    python bootstrap.py --portable

The script does not rely on the repository's top-level CMakeLists.txt.  It
writes a small cross-platform CMake project below .bootstrap/ and builds the
same cpp/mqmas_core.cpp source file.  This makes it work even when the checked-
out CMakeLists.txt still contains compiler flags intended only for GCC/Linux.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import textwrap
import venv
from typing import Mapping, Sequence


MIN_PYTHON = (3, 10)
LIBRARY_NAMES = {
    "windows": "mqmas_core.dll",
    "linux": "libmqmas_core.so",
    "darwin": "libmqmas_core.dylib",
}
ALL_LIBRARY_NAMES = tuple(LIBRARY_NAMES.values())


class BootstrapError(RuntimeError):
    """A user-facing bootstrap failure."""


def log(message: str = "") -> None:
    print(message, flush=True)


def command_text(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in command])
    return shlex.join([str(x) for x in command])


def run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(x) for x in command]
    log(f"+ {command_text(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if check and result.returncode != 0:
        details = f"\n{result.stdout}" if capture and result.stdout else ""
        raise BootstrapError(
            f"Command failed with exit code {result.returncode}: "
            f"{command_text(cmd)}{details}"
        )
    return result


def normalized_system() -> str:
    value = platform.system().strip().lower()
    if value.startswith("win"):
        return "windows"
    if value == "darwin":
        return "darwin"
    if value == "linux":
        return "linux"
    raise BootstrapError(f"Unsupported operating system: {platform.system()!r}")


def normalized_arch(machine: str, bits: int) -> str:
    value = machine.strip().lower().replace(" ", "")
    aliases = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "x64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
        "i386": "x86",
        "i486": "x86",
        "i586": "x86",
        "i686": "x86",
        "x86": "x86",
    }
    if value in aliases:
        arch = aliases[value]
    elif bits == 64 and ("arm" in value or "aarch" in value):
        arch = "arm64"
    elif bits == 64:
        arch = "x86_64"
    elif bits == 32:
        arch = "x86"
    else:
        raise BootstrapError(
            f"Could not map machine architecture {machine!r} ({bits}-bit Python)."
        )

    # ctypes loads into this Python process.  A 32-bit interpreter must receive
    # a 32-bit native library even when the operating system itself is 64-bit.
    if bits == 32 and arch == "x86_64":
        arch = "x86"
    return arch


def cmake_arch(system: str, arch: str) -> str:
    if system == "windows":
        return {"x86_64": "x64", "x86": "Win32", "arm64": "ARM64"}[arch]
    if system == "darwin":
        return {"x86_64": "x86_64", "arm64": "arm64", "x86": "i386"}[arch]
    return arch


def venv_python(venv_dir: Path, system: str) -> Path:
    if system == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_bin_dir(venv_dir: Path, system: str) -> Path:
    return venv_dir / ("Scripts" if system == "windows" else "bin")


def prepend_path(env: dict[str, str], directory: Path) -> None:
    old = env.get("PATH", "")
    env["PATH"] = str(directory) + (os.pathsep + old if old else "")


def which(name: str, env: Mapping[str, str]) -> str | None:
    return shutil.which(name, path=env.get("PATH"))


def repository_root(script_path: Path) -> Path:
    root = script_path.resolve().parent
    required = [
        root / "cpp" / "mqmas_core.cpp",
        root / "python" / "mqmas" / "bindings.py",
        root / "requirements.txt",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise BootstrapError(
            "bootstrap.py must be placed in the MQMAS_OC_Python repository root. "
            f"Missing: {', '.join(missing)}"
        )
    return root


def create_environment(
    directory: Path,
    *,
    system_site_packages: bool,
    recreate: bool,
    system: str,
) -> Path:
    if recreate and directory.exists():
        log(f"Removing existing environment: {directory}")
        shutil.rmtree(directory)

    python = venv_python(directory, system)
    if not python.exists():
        log(f"Creating virtual environment: {directory}")
        builder = venv.EnvBuilder(
            with_pip=True,
            clear=False,
            symlinks=(system != "windows"),
            system_site_packages=system_site_packages,
        )
        try:
            builder.create(directory)
        except Exception as exc:
            hint = (
                " On Debian/Ubuntu, install the matching python3-venv package."
                if system == "linux"
                else ""
            )
            raise BootstrapError(f"Could not create virtual environment {directory}: {exc}.{hint}") from exc
    else:
        log(f"Using existing virtual environment: {directory}")
    return python


def install_python_dependencies(
    python: Path,
    root: Path,
    env: Mapping[str, str],
    *,
    skip: bool,
) -> None:
    if skip:
        log("Skipping Python dependency installation.")
        return
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], env=env)
    run([python, "-m", "pip", "install", "-r", root / "requirements.txt"], env=env)



def inspect_python_interpreter(python: Path, env: Mapping[str, str]) -> dict[str, object]:
    code = (
        "import json, platform, struct, sys; "
        "print(json.dumps({'machine': platform.machine(), "
        "'bits': struct.calcsize('P')*8, "
        "'version': list(sys.version_info[:3])}))"
    )
    result = run([python, "-c", code], env=env, capture=True)
    try:
        return json.loads((result.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise BootstrapError(f"Could not inspect virtual-environment Python: {result.stdout!r}") from exc


def parse_numeric_version(text: str) -> tuple[int, ...]:
    for token in text.replace("\n", " ").split():
        if token and token[0].isdigit():
            parts: list[int] = []
            for part in token.split("."):
                digits = "".join(ch for ch in part if ch.isdigit())
                if not digits:
                    break
                parts.append(int(digits))
            if parts:
                return tuple(parts)
    return ()


def ensure_cmake(
    python: Path,
    env: dict[str, str],
    *,
    allow_install: bool,
) -> str:
    found = which("cmake", env)
    if found:
        result = run([found, "--version"], env=env, capture=True, check=False)
        version = parse_numeric_version(result.stdout or "")
        if result.returncode == 0 and version >= (3, 18):
            return found
        log(f"CMake {'.'.join(map(str, version)) if version else 'of unknown version'} is too old.")

    if allow_install:
        log("Installing CMake >= 3.18 into the virtual environment.")
        run([python, "-m", "pip", "install", "--upgrade", "cmake>=3.18"], env=env)
        found = which("cmake", env)
        if found:
            result = run([found, "--version"], env=env, capture=True, check=False)
            version = parse_numeric_version(result.stdout or "")
            if result.returncode == 0 and version >= (3, 18):
                return found

    raise BootstrapError(
        "CMake 3.18 or newer was not found. Install a current CMake release or "
        "rerun without --no-build-tool-install."
    )

def ensure_tool(
    tool: str,
    package: str,
    python: Path,
    env: dict[str, str],
    *,
    allow_install: bool,
    required: bool,
) -> str | None:
    found = which(tool, env)
    if found:
        return found
    if allow_install:
        log(f"{tool} was not found; installing the {package!r} Python package into the environment.")
        run([python, "-m", "pip", "install", package], env=env)
        found = which(tool, env)
        if found:
            return found
    if required:
        raise BootstrapError(
            f"Required build tool {tool!r} was not found. Install it or rerun without "
            "--no-build-tool-install so the script may install its Python package."
        )
    return None


def find_vswhere() -> str | None:
    candidates: list[Path] = []
    for key in ("ProgramFiles(x86)", "ProgramFiles"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    found = shutil.which("vswhere.exe")
    if found:
        candidates.insert(0, Path(found))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def visual_studio_generator(env: Mapping[str, str]) -> str | None:
    vswhere = find_vswhere()
    if not vswhere:
        return None
    result = run(
        [
            vswhere,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-format",
            "json",
            "-utf8",
        ],
        env=env,
        capture=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        records = json.loads(result.stdout)
        version = str(records[0].get("installationVersion", "")) if records else ""
        major = int(version.split(".", 1)[0])
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return None
    if major >= 17:
        return "Visual Studio 17 2022"
    if major == 16:
        return "Visual Studio 16 2019"
    return None


def select_generator_and_compiler(
    *,
    system: str,
    arch: str,
    env: Mapping[str, str],
    ninja: str | None,
    requested_generator: str | None,
) -> tuple[str, list[str], str | None]:
    """Return generator, extra CMake configure arguments, and compiler path."""
    if requested_generator:
        generator = requested_generator
        extra: list[str] = []
        if generator.startswith("Visual Studio"):
            extra.extend(["-A", cmake_arch(system, arch)])
        if system == "darwin":
            extra.append(f"-DCMAKE_OSX_ARCHITECTURES={cmake_arch(system, arch)}")
        return generator, extra, None

    if system == "windows":
        vs_generator = visual_studio_generator(env)
        if vs_generator:
            return vs_generator, ["-A", cmake_arch(system, arch)], None

        compiler = (
            os.environ.get("CXX")
            or which("cl.exe", env)
            or which("clang-cl.exe", env)
            or which("clang++.exe", env)
            or which("g++.exe", env)
        )
        if compiler and ninja:
            extra = []
            # CMake identifies cl/clang-cl correctly from the environment.
            if Path(compiler).name.lower() not in {"cl.exe", "clang-cl.exe"}:
                extra.append(f"-DCMAKE_CXX_COMPILER={Path(compiler).as_posix()}")
            return "Ninja", extra, compiler
        if compiler and Path(compiler).name.lower() == "cl.exe":
            return "NMake Makefiles", [], compiler
        if compiler and "g++" in Path(compiler).name.lower():
            return "MinGW Makefiles", [f"-DCMAKE_CXX_COMPILER={Path(compiler).as_posix()}"], compiler

        raise BootstrapError(
            "No Windows C++20 toolchain was found. Install Visual Studio 2022 Build Tools "
            "with the 'Desktop development with C++' workload, or put a compatible "
            "clang++/g++ toolchain on PATH. The Visual Studio x64/Win32/ARM64 target is "
            "selected automatically to match the Python interpreter."
        )

    compiler = os.environ.get("CXX") or which("c++", env) or which("g++", env) or which("clang++", env)
    if not compiler:
        if system == "darwin":
            hint = "Install the Xcode command-line tools with: xcode-select --install"
        else:
            hint = "Install g++ (for example, sudo apt install build-essential)."
        raise BootstrapError(f"No C++20 compiler was found. {hint}")

    extra = [f"-DCMAKE_CXX_COMPILER={Path(compiler).as_posix()}"]
    if system == "darwin":
        extra.append(f"-DCMAKE_OSX_ARCHITECTURES={cmake_arch(system, arch)}")
    generator = "Ninja" if ninja else "Unix Makefiles"
    return generator, extra, compiler


def generated_cmake_text(source: Path, output: Path, arch: str) -> str:
    source_text = source.resolve().as_posix()
    output_text = output.resolve().as_posix()
    return textwrap.dedent(
        f"""
        cmake_minimum_required(VERSION 3.18)
        project(mqmas_core LANGUAGES CXX)

        include(CheckCXXCompilerFlag)
        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        set(CMAKE_CXX_EXTENSIONS OFF)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)

        option(MQMAS_NATIVE_OPTIMIZATION "Enable host-specific CPU optimisation" ON)
        set(MQMAS_TARGET_ARCH "{arch}" CACHE STRING "Normalised target architecture")

        add_library(mqmas_core SHARED "{source_text}")
        set_target_properties(mqmas_core PROPERTIES
            OUTPUT_NAME mqmas_core
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            RUNTIME_OUTPUT_DIRECTORY "{output_text}"
            LIBRARY_OUTPUT_DIRECTORY "{output_text}"
            ARCHIVE_OUTPUT_DIRECTORY "{output_text}"
        )

        foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            set_target_properties(mqmas_core PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY_${{_cfg}} "{output_text}"
                LIBRARY_OUTPUT_DIRECTORY_${{_cfg}} "{output_text}"
                ARCHIVE_OUTPUT_DIRECTORY_${{_cfg}} "{output_text}"
            )
        endforeach()

        if(MSVC)
            target_compile_options(mqmas_core PRIVATE /O2 /EHsc /fp:fast)
        else()
            target_compile_options(mqmas_core PRIVATE -O3)
            check_cxx_compiler_flag("-ffast-math" MQMAS_HAS_FAST_MATH)
            if(MQMAS_HAS_FAST_MATH)
                target_compile_options(mqmas_core PRIVATE -ffast-math)
            endif()

            if(MQMAS_NATIVE_OPTIMIZATION)
                if(MQMAS_TARGET_ARCH STREQUAL "arm64")
                    check_cxx_compiler_flag("-mcpu=native" MQMAS_HAS_MCPU_NATIVE)
                    if(MQMAS_HAS_MCPU_NATIVE)
                        target_compile_options(mqmas_core PRIVATE -mcpu=native)
                    endif()
                else()
                    check_cxx_compiler_flag("-march=native" MQMAS_HAS_MARCH_NATIVE)
                    if(MQMAS_HAS_MARCH_NATIVE)
                        target_compile_options(mqmas_core PRIVATE -march=native)
                    endif()
                endif()
            endif()
        endif()

        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(mqmas_core PRIVATE OpenMP::OpenMP_CXX)
            message(STATUS "MQMAS OpenMP: enabled")
        else()
            message(STATUS "MQMAS OpenMP: not found; building the serial core")
        endif()

        message(STATUS "MQMAS target architecture: ${{MQMAS_TARGET_ARCH}}")
        message(STATUS "MQMAS C++ compiler: ${{CMAKE_CXX_COMPILER_ID}} ${{CMAKE_CXX_COMPILER_VERSION}}")
        """
    ).strip() + "\n"


def library_name(system: str) -> str:
    return LIBRARY_NAMES[system]


def find_built_library(output_dir: Path, build_dir: Path, system: str) -> Path:
    name = library_name(system)
    direct = output_dir / name
    if direct.exists():
        return direct
    matches = list(output_dir.rglob(name)) + list(build_dir.rglob(name))
    if not matches:
        raise BootstrapError(
            f"The native build completed but {name!r} was not found below "
            f"{output_dir} or {build_dir}."
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def copy_mingw_runtime_dlls(compiler: str | None, destination: Path) -> list[str]:
    if not compiler or "g++" not in Path(compiler).name.lower():
        return []
    copied: list[str] = []
    compiler_dir = Path(compiler).resolve().parent
    candidates = (
        "libgomp-1.dll",
        "libstdc++-6.dll",
        "libwinpthread-1.dll",
        "libgcc_s_seh-1.dll",
        "libgcc_s_sjlj-1.dll",
        "libgcc_s_dw2-1.dll",
    )
    for name in candidates:
        source = compiler_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
            copied.append(name)
    return copied


def install_library(
    built: Path,
    root: Path,
    *,
    system: str,
    compiler: str | None,
) -> tuple[Path, list[str]]:
    destination_dir = root / "build"
    destination_dir.mkdir(parents=True, exist_ok=True)

    # The current bindings search .so before .dylib before .dll.  Remove stale
    # binaries from other operating systems so the loader cannot select one.
    target_name = library_name(system)
    for name in ALL_LIBRARY_NAMES:
        path = destination_dir / name
        if name != target_name and path.exists():
            log(f"Removing incompatible stale library: {path}")
            path.unlink()

    target = destination_dir / target_name
    shutil.copy2(built, target)
    runtimes: list[str] = []
    if system == "windows":
        runtimes = copy_mingw_runtime_dlls(compiler, destination_dir)
    return target, runtimes


def install_repo_pth(python: Path, root: Path, env: Mapping[str, str]) -> Path:
    code = (
        "import site, pathlib; "
        "p=pathlib.Path(site.getsitepackages()[0])/'mqmas_repo.pth'; "
        f"p.write_text({str((root / 'python').resolve())!r}+'\\n', encoding='utf-8'); "
        "print(p)"
    )
    result = run([python, "-c", code], env=env, capture=True)
    path = Path((result.stdout or "").strip().splitlines()[-1])
    return path


def validate_native_load(python: Path, library: Path, env: Mapping[str, str]) -> None:
    code = (
        "import ctypes, pathlib; "
        f"p=pathlib.Path({str(library.resolve())!r}); "
        "lib=ctypes.CDLL(str(p)); "
        "getattr(lib, 'mqmas_create_d4'); "
        "getattr(lib, 'mqmas_eval_pair_d6'); "
        "print('Loaded', p)"
    )
    test_env = dict(env)
    prepend_path(test_env, library.parent)
    run([python, "-c", code], env=test_env)


def compiler_from_cache(build_dir: Path) -> dict[str, str]:
    cache = build_dir / "CMakeCache.txt"
    values: dict[str, str] = {}
    if not cache.exists():
        return values
    keys = {
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "OpenMP_CXX_FLAGS",
    }
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        left, value = line.split("=", 1)
        key = left.split(":", 1)[0]
        if key in keys:
            values[key] = value
    return values


def activation_instructions(venv_dir: Path, system: str) -> list[str]:
    if system == "windows":
        return [
            f"PowerShell:  {venv_dir}\\Scripts\\Activate.ps1",
            f"cmd.exe:     {venv_dir}\\Scripts\\activate.bat",
        ]
    return [f"bash/zsh:    source {shlex.quote(str(venv_dir / 'bin' / 'activate'))}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an architecture-matched Python environment and build mqmas_core.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--venv", type=Path, help="Virtual-environment directory.")
    parser.add_argument(
        "--arch",
        choices=("auto", "x86_64", "x86", "arm64"),
        default="auto",
        help="Native target architecture. 'auto' matches the running Python interpreter.",
    )
    parser.add_argument("--generator", help="Explicit CMake generator override.")
    parser.add_argument("--build-type", default="Release", choices=("Release", "RelWithDebInfo", "Debug"))
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--clean", action="store_true", help="Remove the architecture-specific native build tree first.")
    parser.add_argument("--recreate-venv", action="store_true", help="Delete and recreate the virtual environment.")
    parser.add_argument("--skip-deps", action="store_true", help="Do not install requirements.txt.")
    parser.add_argument("--skip-tests", action="store_true", help="Build and load the library but do not run pytest.")
    parser.add_argument(
        "--no-build-tool-install",
        action="store_true",
        help="Do not install CMake/Ninja into the virtual environment when missing.",
    )
    parser.add_argument(
        "--system-site-packages",
        action="store_true",
        help="Allow the virtual environment to see packages from the base interpreter.",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Disable -march=native/-mcpu=native for a more portable binary.",
    )
    return parser.parse_args()


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        raise BootstrapError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required; "
            f"this interpreter is {platform.python_version()}."
        )

    args = parse_args()
    root = repository_root(Path(__file__))
    system = normalized_system()
    bits = struct.calcsize("P") * 8
    detected_arch = normalized_arch(platform.machine(), bits)
    arch = detected_arch if args.arch == "auto" else args.arch

    if args.arch != "auto" and arch != detected_arch:
        raise BootstrapError(
            f"Requested architecture {arch!r} does not match this Python process "
            f"({detected_arch}, {bits}-bit). A ctypes library must match Python. "
            "Run the matching Python interpreter instead."
        )

    tag = f"{system}-{arch}-py{bits}"
    venv_dir = (args.venv or (root / f".venv-{system}-{arch}")).resolve()

    log("MQMAS bootstrap")
    log("---------------")
    log(f"Repository:          {root}")
    log(f"Operating system:    {platform.system()} {platform.release()}")
    log(f"Machine reported:    {platform.machine()}")
    log(f"Python:              {platform.python_version()} ({bits}-bit)")
    log(f"Native target:       {arch}")
    log(f"Build tag:           {tag}")
    log()

    python = create_environment(
        venv_dir,
        system_site_packages=args.system_site_packages,
        recreate=args.recreate_venv,
        system=system,
    )

    env = os.environ.copy()
    prepend_path(env, venv_bin_dir(venv_dir, system))
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    interpreter_info = inspect_python_interpreter(python, env)
    venv_bits = int(interpreter_info.get("bits", 0))
    venv_machine = str(interpreter_info.get("machine", ""))
    venv_arch = normalized_arch(venv_machine, venv_bits)
    if venv_bits != bits or venv_arch != arch:
        raise BootstrapError(
            f"Existing environment uses {venv_arch}/{venv_bits}-bit Python, but this "
            f"build requires {arch}/{bits}-bit Python. Rerun with --recreate-venv."
        )

    install_python_dependencies(python, root, env, skip=args.skip_deps)

    allow_tool_install = not args.no_build_tool_install
    cmake = ensure_cmake(
        python,
        env,
        allow_install=allow_tool_install,
    )
    ninja = ensure_tool(
        "ninja",
        "ninja",
        python,
        env,
        allow_install=allow_tool_install,
        required=False,
    )

    generator, generator_args, compiler = select_generator_and_compiler(
        system=system,
        arch=arch,
        env=env,
        ninja=ninja,
        requested_generator=args.generator,
    )

    bootstrap_root = root / ".bootstrap"
    cmake_source_dir = bootstrap_root / "cmake" / tag
    build_dir = bootstrap_root / "build" / f"{tag}-{generator.replace(' ', '_')}"
    output_dir = bootstrap_root / "output" / tag

    if args.clean:
        for path in (build_dir, output_dir):
            if path.exists():
                log(f"Removing {path}")
                shutil.rmtree(path)

    cmake_source_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_cmake = cmake_source_dir / "CMakeLists.txt"
    generated_cmake.write_text(
        generated_cmake_text(root / "cpp" / "mqmas_core.cpp", output_dir, arch),
        encoding="utf-8",
    )

    configure_command: list[str] = [
        cmake,
        "-S",
        str(cmake_source_dir),
        "-B",
        str(build_dir),
        "-G",
        generator,
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DMQMAS_NATIVE_OPTIMIZATION={'OFF' if args.portable else 'ON'}",
        *generator_args,
    ]

    log()
    log(f"CMake generator:     {generator}")
    if compiler:
        log(f"Detected compiler:   {compiler}")
    log(f"Native build tree:   {build_dir}")
    log()

    run(configure_command, cwd=root, env=env)
    run(
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            args.build_type,
            "--parallel",
            str(max(1, args.jobs)),
        ],
        cwd=root,
        env=env,
    )

    built = find_built_library(output_dir, build_dir, system)
    installed, runtime_dlls = install_library(
        built,
        root,
        system=system,
        compiler=compiler,
    )
    pth = install_repo_pth(python, root, env)

    log()
    log(f"Installed native core: {installed}")
    if runtime_dlls:
        log(f"Copied MinGW runtime DLLs: {', '.join(runtime_dlls)}")
    log(f"Installed import path: {pth}")

    validate_native_load(python, installed, env)

    if not args.skip_tests:
        test_env = dict(env)
        prepend_path(test_env, installed.parent)
        test_env.setdefault("OMP_NUM_THREADS", str(max(1, min(args.jobs, os.cpu_count() or 1))))
        test_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        test_env.setdefault("MKL_NUM_THREADS", "1")
        run([python, "-m", "pytest", "-q"], cwd=root, env=test_env)

    cache_info = compiler_from_cache(build_dir)
    build_info = {
        "system": system,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "target_arch": arch,
        "python": platform.python_version(),
        "python_bits": bits,
        "python_executable": str(python),
        "venv": str(venv_dir),
        "generator": generator,
        "build_type": args.build_type,
        "portable": bool(args.portable),
        "library": str(installed),
        "runtime_dlls": runtime_dlls,
        "cmake": cmake,
        "cmake_cache": cache_info,
    }
    info_path = root / "build" / "mqmas_build_info.json"
    info_path.write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")

    log()
    log("Bootstrap completed successfully.")
    log(f"Build information: {info_path}")
    log("Activate the environment with:")
    for line in activation_instructions(venv_dir, system):
        log(f"  {line}")
    log()
    log("The repository's python/ directory is installed through a .pth file, so no PYTHONPATH setting is required.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
