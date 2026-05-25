@echo off
set WHISPER_MODEL=%1
if "%WHISPER_MODEL%"=="" set WHISPER_MODEL=base.en
cd /d "%~dp0"
python akashvani.py
