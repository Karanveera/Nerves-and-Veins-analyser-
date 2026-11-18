#!/bin/bash
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3 from python.org"; read -p "Press Enter"; exit 1; }
if [ ! -d .venv ]; then
  echo "First run: setting things up, this takes a minute..."
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip -q && pip install -r requirements.txt
else
  source .venv/bin/activate
fi
streamlit run app.py
