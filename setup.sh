#!/bin/bash
set -e

echo "Setting up codebase..."

if [ ! -d .venv ]; then
  echo "No virtual environment found. Creating..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e .
  echo "Setup complete."
else
  echo "Virtual environment already exists, check for updates."
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e .
  echo "Setup complete."
fi


