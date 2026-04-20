@echo off
echo Creating virtual environment...
python -m venv projEnv

echo Activating environment...
call projEnv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Done! Environment ready.
pause