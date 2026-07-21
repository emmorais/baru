@echo off
REM ===========================================================================
REM Windows build helper for baru.
REM
REM The vendored SEAL/CKKS backend (third_party/sealy) is a C++ library built
REM via cmake + MSVC, and bindgen's libclang needs the Windows SDK include
REM paths.  A plain `cargo build` from a bare shell therefore fails.  This
REM helper enters the Visual Studio developer environment and adds the two
REM extras SEAL needs:
REM   * NETFXSDK's um dir on INCLUDE, for CorError.h (SEAL's C wrapper includes
REM     it; the modern Windows SDK no longer ships it).
REM   * NASM on PATH (needed by aws-lc-rs when building --features pq-tls).
REM
REM Usage:  build-windows.cmd <cargo args...>
REM   build-windows.cmd build
REM   build-windows.cmd test --lib
REM   build-windows.cmd build --features pq-tls
REM ===========================================================================
setlocal

REM --- locate Visual Studio (vswhere), fall back to VS2022 Community ----------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
)
if not defined VSINSTALL set "VSINSTALL=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
set "VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo [build-windows] ERROR: vcvars64.bat not found at "%VCVARS%".
  echo [build-windows] Install Visual Studio / Build Tools with the "Desktop development with C++" workload.
  exit /b 1
)
call "%VCVARS%" >nul
if errorlevel 1 (
  echo [build-windows] ERROR: vcvars64.bat failed.
  exit /b 1
)

REM --- CorError.h (ships with the .NET Framework SDK / NETFXSDK) --------------
set "NETFX=%ProgramFiles(x86)%\Windows Kits\NETFXSDK\4.8\Include\um"
if exist "%NETFX%\CorError.h" (
  set "INCLUDE=%NETFX%;%INCLUDE%"
) else (
  echo [build-windows] WARN: CorError.h not found under NETFXSDK 4.8; SEAL may fail to build.
  echo [build-windows]       Install an older Windows SDK, e.g.:
  echo [build-windows]       winget install Microsoft.WindowsSDK.10.0.18362
)

REM --- NASM for aws-lc-rs (only needed for --features pq-tls) -----------------
if exist "%LOCALAPPDATA%\bin\NASM\nasm.exe" set "PATH=%LOCALAPPDATA%\bin\NASM;%PATH%"

echo [build-windows] VS: %VSINSTALL%
cargo %*
