@echo off
python -m pip install --upgrade pip
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --prefer-binary -r requirements.txt
if %ERRORLEVEL% neq 0 (
  echo Installation failed.
  exit /b %ERRORLEVEL%
)
echo Installation complete.
