@echo off
REM Tag Classifier Server starten
REM Standardport: 8766

echo Aktiviere virtuelles Environment...
call venv\Scripts\activate.bat

echo Starte Tag Classifier Server...
python server.py %*

pause
