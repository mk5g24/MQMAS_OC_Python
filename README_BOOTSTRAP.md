# MQMAS cross-platform bootstrap

Place `bootstrap.py` in the root of the `MQMAS_OC_Python` repository, next to
`requirements.txt`, `cpp/`, and `python/`.

## Normal use

### Windows PowerShell

```powershell
py -3.11 .\bootstrap.py --clean
```

Python 3.10 or newer may be used. The script targets the architecture of the
Python interpreter, because the DLL loaded through `ctypes` must have the same
architecture as Python. Examples are x64 Python -> x64 DLL and ARM64 Python ->
ARM64 DLL.

A C++ compiler is still required. The recommended Windows installation is
Visual Studio 2022 Build Tools with the **Desktop development with C++**
workload. The script detects Visual Studio through `vswhere`, selects `x64`,
`Win32`, or `ARM64`, and enables automatic symbol export for the DLL.

### Linux

```bash
python3 bootstrap.py --clean
```

Install a C++20 compiler first, for example:

```bash
sudo apt install build-essential
```

### macOS

```bash
python3 bootstrap.py --clean
```

Install Apple's command-line tools first if necessary:

```bash
xcode-select --install
```

## What the script does

1. Detects Windows, Linux, or macOS and normalises the Python process
   architecture to `x86_64`, `x86`, or `arm64`.
2. Creates an isolated environment named `.venv-<system>-<architecture>`.
3. Installs `requirements.txt`.
4. Uses an existing CMake 3.18+ installation, or installs CMake into the virtual
   environment when it is absent or too old.
5. Uses Visual Studio on Windows when available; otherwise it can use Ninja
   with a compiler already on `PATH`. On Linux/macOS it selects Ninja or Unix
   Makefiles.
6. Generates a private cross-platform CMake project under `.bootstrap/`. This
   avoids the compiler-specific flags in the repository's original
   `CMakeLists.txt`.
7. Builds `cpp/mqmas_core.cpp` with C++20 and OpenMP when available.
8. Copies the native library to the location used by the current bindings:
   - Windows: `build/mqmas_core.dll`
   - Linux: `build/libmqmas_core.so`
   - macOS: `build/libmqmas_core.dylib`
9. Removes stale native libraries for other operating systems, preventing a
   Windows process from attempting to load a Linux `.so`.
10. Adds `python/` to the virtual environment through an `.pth` file, verifies
    the exported C API with `ctypes`, and runs the test suite.
11. Writes the detected toolchain information to
    `build/mqmas_build_info.json`.

## Useful options

```text
--clean                Reconfigure and rebuild the native core from scratch.
--recreate-venv        Delete and recreate the virtual environment.
--skip-tests           Do not run pytest after building.
--skip-deps            Do not install requirements.txt.
--portable             Disable -march=native/-mcpu=native.
--jobs N               Select the number of parallel build jobs.
--generator NAME       Override automatic CMake-generator selection.
--venv PATH            Choose a different virtual-environment directory.
```

For a fully fresh installation:

```powershell
py -3.11 .\bootstrap.py --clean --recreate-venv
```

Afterwards, activate the environment using the command printed at the end of
the bootstrap. No `PYTHONPATH` setting is required.

## Suggested `.gitignore` entries

```gitignore
.venv-*/
.bootstrap/
build/
__pycache__/
.pytest_cache/
```
