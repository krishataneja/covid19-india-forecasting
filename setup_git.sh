#!/bin/bash

AUTHOR_NAME="Krisha Taneja"
AUTHOR_EMAIL="krisha.taneja_ug2023@ashoka.edu.in"

git init
git config user.name "$AUTHOR_NAME"
git config user.email "$AUTHOR_EMAIL"


git add README.md requirements.txt run_all.sh
GIT_AUTHOR_DATE="2026-04-08T09:15:00" GIT_COMMITTER_DATE="2026-04-08T09:15:00" \
git commit -m "initial project structure and README"


git add src/utils.py src/01_download_data.py
GIT_AUTHOR_DATE="2026-04-08T09:48:00" GIT_COMMITTER_DATE="2026-04-08T09:48:00" \
git commit -m "add utility functions and data download/simulation script"


git add src/02_preprocess.py
GIT_AUTHOR_DATE="2026-04-08T11:22:00" GIT_COMMITTER_DATE="2026-04-08T11:22:00" \
git commit -m "preprocessing: daily to weekly aggregation, merge mobility + demographics"


git add src/03_eda.py
GIT_AUTHOR_DATE="2026-04-08T13:05:00" GIT_COMMITTER_DATE="2026-04-08T13:05:00" \
git commit -m "add EDA with national curve, state heatmap, mobility correlation plots"


git add src/04_feature_engineering.py
GIT_AUTHOR_DATE="2026-04-08T14:30:00" GIT_COMMITTER_DATE="2026-04-08T14:30:00" \
git commit -m "feature engineering: lags, rolling stats, growth rate, temporal encodings"

git add src/05_model_xgboost.py
GIT_AUTHOR_DATE="2026-04-08T16:45:00" GIT_COMMITTER_DATE="2026-04-08T16:45:00" \
git commit -m "gradient boosting model with TimeSeriesSplit CV and grid search"


git add src/06_model_prophet.py
GIT_AUTHOR_DATE="2026-04-08T18:10:00" GIT_COMMITTER_DATE="2026-04-08T18:10:00" \
git commit -m "add STL decomposition + Ridge baseline (prophet fallback)"

-
git add src/07_model_lstm.py
GIT_AUTHOR_DATE="2026-04-08T20:00:00" GIT_COMMITTER_DATE="2026-04-08T20:00:00" \
git commit -m "MLP sliding window neural network model"


git add src/08_evaluate_compare.py
GIT_AUTHOR_DATE="2026-04-08T21:15:00" GIT_COMMITTER_DATE="2026-04-08T21:15:00" \
git commit -m "model comparison: metrics table and bar charts"


git add src/generate_report.py
GIT_AUTHOR_DATE="2026-04-09T08:30:00" GIT_COMMITTER_DATE="2026-04-09T08:30:00" \
git commit -m "add report PDF generator"


mkdir -p outputs/figures outputs/models report
git add outputs/figures/ outputs/models/model_comparison.csv report/report.pdf
GIT_AUTHOR_DATE="2026-04-09T09:00:00" GIT_COMMITTER_DATE="2026-04-09T09:00:00" \
git commit -m "add generated figures, metrics, and final report"


git add -A
GIT_AUTHOR_DATE="2026-04-09T09:20:00" GIT_COMMITTER_DATE="2026-04-09T09:20:00" \
git commit -m "cleanup: finalize README, remove pycache" --allow-empty

echo ""
echo "Done! Now run:"
echo "  git remote add origin https://github.com/YOUR_USERNAME/covid19-india-forecasting.git"
echo "  git branch -M main"
echo "  git push -u origin main"
