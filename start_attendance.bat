@echo off
cd /d "%~dp0"
set KIOSK_KEY=change-this-key

rem Use the project's virtual environment if there is one
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

python app.py
pause