#!/usr/bin/env bash

python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --prefer-binary -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Installation failed."
  exit 1
fi
printf "Installation complete.\n"
