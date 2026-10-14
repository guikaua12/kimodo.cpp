@echo off
rem Kimodo web UI: http://127.0.0.1:8094  (close this window to stop the server)
cd /d "%~dp0"
set "PATH=%~dp0build-win\bin;%PATH%"
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8094"
"%~dp0build-win\kimodo-demo.exe" -addr 127.0.0.1:8094 -generator "%~dp0build-win\kmd-generate.exe"
