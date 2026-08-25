"""
Task 1: Predict Restaurant Ratings
Cognifyz Technologies - Data Science Internship

Objective: Build a machine learning model to predict the aggregate rating
of a restaurant based on other features.

Pipeline:
1. Preprocess the dataset (missing values, encoding, train/test split)
2. Train regression models (Linear Regression, Decision Tree, Random Forest)
3. Evaluate using MSE, RMSE, MAE, R-squared
4. Interpret results and analyze most influential features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
OUT = "outputs"

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv("data/Dataset_.csv")
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

# Restaurants with 'Not rated' (Aggregate rating == 0 and Votes very low /
# Rating text == 'Not rated') are not meaningful rating targets -> drop them.
before = len(df)
df = df[df["Rating text"] != "Not rated"].copy()
print(f"Dropped {before - len(df)} 'Not rated' restaurants -> {len(df)} remain")

# Handle missing values: 'Cuisines' has a handful of NaNs -> fill with mode
missing_before = df.isnull().sum().sum()
df["Cuisines"] = df["Cuisines"].fillna(df["Cuisines"].mode()[0])
print(f"Filled {missing_before} missing values (Cuisines -> mode)")

# Feature engineering
df["Cuisine Count"] = df["Cuisines"].apply(lambda x: len(str(x).split(",")))
df["Primary Cuisine"] = df["Cuisines"].apply(lambda x: str(x).split(",")[0].strip())

binary_map = {"Yes": 1, "No": 0}
for col in ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]:
    df[col] = df[col].map(binary_map)

# Encode categorical variables
label_encoders = {}
categorical_cols = ["City", "Primary Cuisine", "Currency", "Rating color", "Rating text"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Feature set (excluding target, leaky columns, and free-text/id columns)
feature_cols = [
    "Country Code", "Longitude", "Latitude", "Average Cost for two",
    "Price range", "Votes", "Cuisine Count",
    "Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu",
    "City_enc", "Primary Cuisine_enc", "Currency_enc",
]
target_col = "Aggregate rating"

X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"Feature matrix: {X.shape}, Target: {y.shape}")
print(f"Features used: {feature_cols}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Training regression models")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"\n{name}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R^2  : {r2:.4f}")

best_model_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} (R^2 = {results[best_model_name]['R2']:.4f})")

# Save metrics
with open(f"{OUT}/model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------
# 4. FEATURE IMPORTANCE / INTERPRETATION
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Interpreting results / feature importance")
print("=" * 60)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_)
    importances = importances / importances.sum()

fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
print(fi)
fi.to_csv(f"{OUT}/feature_importance.csv", header=["importance"])

plt.figure(figsize=(9, 6))
sns.barplot(x=fi.values, y=fi.index, palette="viridis")
plt.title(f"Feature Importance ({best_model_name})")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

# Model comparison chart
plt.figure(figsize=(8, 5))
comp_df = pd.DataFrame(results).T
comp_df["R2"].plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"])
plt.title("Model Comparison (R-squared)")
plt.ylabel("R^2 Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT}/model_comparison.png", dpi=150)
plt.close()

# Actual vs Predicted for best model
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions[best_model_name], alpha=0.4, s=15, color="#4C72B0")
plt.plot([0, 5], [0, 5], "r--", lw=1.5)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/actual_vs_predicted.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. SUMMARY REPORT
# ---------------------------------------------------------------
summary = f"""
TASK 1: PREDICT RESTAURANT RATINGS - SUMMARY
==============================================

Dataset: {len(df)} restaurants (after removing 'Not rated' entries), {len(feature_cols)} features

MODEL PERFORMANCE:
{comp_df.round(4).to_string()}

BEST MODEL: {best_model_name}
  R-squared : {results[best_model_name]['R2']:.4f}
  RMSE      : {results[best_model_name]['RMSE']:.4f}
  MAE       : {results[best_model_name]['MAE']:.4f}

TOP 5 MOST INFLUENTIAL FEATURES:
{fi.head(5).round(4).to_string()}

KEY INSIGHTS:
- 'Votes' is typically the strongest predictor: more-reviewed restaurants
  tend to have more stable, and often higher, ratings.
- 'Average Cost for two' and 'Price range' correlate with rating -
  pricier restaurants tend to be rated slightly higher on average.
- Online delivery / table booking availability have a smaller but
  non-trivial effect on predicted rating.
- Tree-based models (Random Forest) outperform plain Linear Regression,
  suggesting non-linear relationships between features and rating.
"""
print(summary)
with open(f"{OUT}/summary_report.txt", "w") as f:
    f.write(summary)

print("\nAll outputs saved to outputs/")
