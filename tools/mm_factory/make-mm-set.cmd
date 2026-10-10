@echo off
rem One-click motion-matching set generator (prototype).
rem Double-click to be asked for a style, or run: make-mm-set.cmd zombie
rem Output: factory-output\<style>\ (GLBs, SOMA\ mesh GLBs, manifest.json, report.md)
cd /d "%~dp0..\.."
set "STYLE=%~1"
if "%STYLE%"=="" set /p "STYLE=Style (neutral, zombie, rifle, crouch, old, goblin) [zombie]: "
if "%STYLE%"=="" set "STYLE=zombie"
python -I tools\mm_factory\factory.py --style %STYLE% --soma
if errorlevel 1 (
  echo.
  echo Generation failed - see factory-output\%STYLE%\_work\factory.log
  pause
  exit /b 1
)
start "" "%cd%\factory-output\%STYLE%"
pause
