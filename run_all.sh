#!/bin/bash
# run_all.sh — Execute the complete COVID-19 forecasting pipeline
set -e

echo "============================================"
echo "COVID-19 India Forecasting Pipeline"
echo "============================================"

cd "$(dirname "$0")"

echo ""
echo "[1/8] Data preparation ..."
python src/01_download_data.py --simulate

echo ""
echo "[2/8] Preprocessing ..."
python src/02_preprocess.py

echo ""
echo "[3/8] Exploratory data analysis ..."
python src/03_eda.py

echo ""
echo "[4/8] Feature engineering ..."
python src/04_feature_engineering.py

echo ""
echo "[5/8] Training XGBoost ..."
python src/05_model_xgboost.py

echo ""
echo "[6/8] Training Prophet ..."
python src/06_model_prophet.py

echo ""
echo "[7/8] Training LSTM ..."
python src/07_model_lstm.py

echo ""
echo "[8/8] Model comparison ..."
python src/08_evaluate_compare.py

echo ""
echo "============================================"
echo "Pipeline complete! Check outputs/ directory."
echo "============================================"
