@echo off
rem Rebuild kimodo.cpp (Vulkan, Release) into build-win. Safe to re-run.
rem Needs Visual Studio or the Build Tools with the C++ workload, the Vulkan
rem SDK (its installer sets VULKAN_SDK) and Go on PATH. Works from a plain
rem prompt (finds Visual Studio through vswhere) or a Developer Command Prompt.
setlocal
cd /d "%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined VCINSTALLDIR if not exist "%VSWHERE%" (echo vswhere.exe not found: install Visual Studio or the Build Tools with the C++ workload & exit /b 1)
if not defined VCINSTALLDIR for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VCINSTALLDIR if not defined VSDIR (echo no Visual Studio install with the C++ tools found & exit /b 1)
if not defined VCINSTALLDIR call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined VULKAN_SDK (echo VULKAN_SDK is not set: install the Vulkan SDK & exit /b 1)
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
where go >nul 2>&1 || (echo go not found on PATH: install Go & exit /b 1)
echo --- toolchain:
where cl & where cmake & where ninja & where glslc & where go
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DKIMODO_ENABLE_VULKAN=ON -DKIMODO_BUILD_TESTS=OFF || exit /b 1
cmake --build build-win || exit /b 1
go build -o build-win\kimodo-demo.exe ./demo || exit /b 1
echo BUILD_OK
