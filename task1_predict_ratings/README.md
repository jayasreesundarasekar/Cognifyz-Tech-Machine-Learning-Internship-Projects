# Task 1: Predict Restaurant Ratings
### Cognifyz Technologies — Data Science Internship

## Objective
Build a machine learning model to predict the aggregate rating of a restaurant based on its other features (location, cost, votes, cuisine, services offered, etc.).

## Contents
```
task1_predict_ratings/
├── data/
│   └── Dataset_.csv              # Raw Zomato restaurant dataset (9,551 rows)
├── outputs/
│   ├── model_metrics.json        # MSE, RMSE, MAE, R² for all 3 models
│   ├── feature_importance.csv    # Ranked feature importances
│   ├── feature_importance.png    # Bar chart of feature importances
│   ├── model_comparison.png      # R² comparison across models
│   ├── actual_vs_predicted.png   # Scatter plot for best model
│   └── summary_report.txt        # Full written summary of results
├── task1_predict_ratings.py      # End-to-end pipeline script
└── README.md
```

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python3 task1_predict_ratings.py
```

## Pipeline steps
1. **Preprocessing**
   - Dropped 2,148 "Not rated" restaurants (rating = 0, essentially unrated, not a true low score)
   - Filled 9 missing `Cuisines` values with the mode
   - Engineered `Cuisine Count` and `Primary Cuisine`
   - Label-encoded categorical variables (City, Primary Cuisine, Currency, etc.)
   - Converted Yes/No service flags to binary
   - 80/20 train-test split (random_state=42)

2. **Model training** — compared 3 regression algorithms:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor

3. **Evaluation** (on held-out test set):

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | 0.183 | 0.428 | 0.338 | 0.409 |
| Decision Tree | 0.134 | 0.367 | 0.270 | 0.565 |
| **Random Forest** | **0.111** | **0.333** | **0.247** | **0.641** |

4. **Interpretation** — Random Forest was the best performer (R² = 0.64).
   Top influential features: **Votes** (dominant, ~55% importance), followed by
   **Longitude/Latitude** (regional rating patterns), **Primary Cuisine**, and
   **Average Cost for two**.

## Key Insights
- Number of votes is by far the strongest signal — restaurants with more reviews tend to have more reliable, generally higher ratings.
- Geographic location matters — certain regions/cities systematically rate higher.
- Cuisine type and price point have moderate influence.
- Whether a restaurant offers table booking, online delivery, etc. has minimal impact on predicted rating.
- Non-linear models (Decision Tree, Random Forest) substantially outperform Linear Regression, indicating the rating relationship isn't purely linear.
