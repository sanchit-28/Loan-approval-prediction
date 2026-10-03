#!/usr/bin/env bash
# Loan Approval Prediction - runner for macOS / Linux
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python src/loan_approval.py
echo "Done. Charts: reports/figures   Metrics: results/results_summary.json"
