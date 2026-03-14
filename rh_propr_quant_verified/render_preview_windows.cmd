@echo off
setlocal
set QT_QPA_PLATFORM=offscreen
python render_preview.py
endlocal
