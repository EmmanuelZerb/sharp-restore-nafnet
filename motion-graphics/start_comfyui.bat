@echo off
cd /d D:\ComfyUI\ComfyUI_windows_portable
echo Starting ComfyUI...
echo.
echo Once loaded, open: http://127.0.0.1:8188
echo.
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build
pause
