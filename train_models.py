"""
train_models.py
---------------------------------------------------------------------------
Petroleum Refinery Yield Prediction System
Trains and compares three regression algorithms (Random Forest, XGBoost,
Support Vector Regressor) on crude oil physicochemical properties to
predict the volumetric yield (%) of five refinery product fractions.

USAGE (in PyCharm or terminal):
    1. cd petro_yield_project
    2. python data/generate_synthetic_data.py      (only needed once, or
                                                      whenever you swap data)
    3. python train_models.py

OUTPUT:
    - model/best_model.pkl       -> the best-performing trained model
    - model/scaler.pkl           -> the fitted StandardScaler
    - model/feature_importance.png
    - model/model_comparison.png
    - Printed MAE / RMSE / R2 for all three models
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "crude_oil_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

NUMERIC_FEATURES = ["api_gravity", "specific_gravity", "pour_point_f",
                     "viscosity_cst", "sulfur_pct", "nitrogen_pct"]
CATEGORICAL_FEATURES = ["geological_formation", "crude_source"]
TARGET_COLUMNS = ["light_gas_yield_pct", "kerosene_yield_pct", "gas_oil_yield_pct",
                   "lubricant_yield_pct", "residual_yield_pct"]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET_COLUMNS]

# ---------------------------------------------------------------------------
# 2. PREPROCESSING (scale numeric, one-hot encode categorical)
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (80:20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------------------------------------------------------------------------
# 4. DEFINE THE THREE CANDIDATE MODELS
#    Random Forest and XGBoost natively support multi-output regression
#    via MultiOutputRegressor (since we predict 5 yield fractions at once).
#    SVR is inherently single-output, so it is wrapped the same way.
# ---------------------------------------------------------------------------
models = {
    "Random Forest Regressor": Pipeline([
        ("prep", preprocessor),
        ("model", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
        )),
    ]),
    "XGBoost Regressor": Pipeline([
        ("prep", preprocessor),
        ("model", MultiOutputRegressor(
            XGBRegressor(n_estimators=200, learning_rate=0.08,
                          max_depth=5, random_state=42, verbosity=0)
        )),
    ]),
    "Support Vector Regressor (SVR)": Pipeline([
        ("prep", preprocessor),
        ("model", MultiOutputRegressor(
            SVR(kernel="rbf", C=20, epsilon=0.5)
        )),
    ]),
}

# ---------------------------------------------------------------------------
# 5. TRAIN, PREDICT, EVALUATE
# ---------------------------------------------------------------------------
results = {}
trained_pipelines = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    trained_pipelines[name] = pipeline

    print(f"\n{name}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R2   : {r2:.4f}")

# ---------------------------------------------------------------------------
# 6. SELECT BEST MODEL (highest R2)
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["R2"])
best_pipeline = trained_pipelines[best_name]
print(f"\n>>> Best-performing model: {best_name} (R2 = {results[best_name]['R2']:.4f})")

joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(TARGET_COLUMNS, os.path.join(MODEL_DIR, "target_columns.pkl"))
joblib.dump(best_name, os.path.join(MODEL_DIR, "best_model_name.pkl"))

# Save a results table for reference / reporting
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison_results.csv"))
print("\nSaved comparison table to model/model_comparison_results.csv")

# ---------------------------------------------------------------------------
# 7. VISUALIZATION 1 — Model comparison bar chart (R2 per model)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.barplot(x=list(results.keys()), y=[results[m]["R2"] for m in results],
            palette=["#4C72B0", "#DD8452", "#55A868"])
plt.ylabel("R\u00B2 Score (test set)")
plt.title("Model Comparison: R\u00B2 Score by Algorithm")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. VISUALIZATION 2 — Feature importance (only meaningful for tree models)
# ---------------------------------------------------------------------------
if "Random Forest" in best_name or "XGBoost" in best_name:
    # Pull feature names after preprocessing
    ohe = best_pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names

    # Average feature importance across the 5 multi-output estimators
    importances = np.mean([
        est.feature_importances_ for est in best_pipeline.named_steps["model"].estimators_
    ], axis=0)

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(10)

    plt.figure(figsize=(7, 5))
    sns.barplot(x="importance", y="feature", data=imp_df, color="#4C72B0")
    plt.title(f"Top 10 Feature Importances ({best_name})")
    plt.xlabel("Average Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved feature_importance.png")
else:
    print("Best model is SVR — skipping tree-based feature importance plot.")

print("\nTraining complete. Run app.py to launch the web interface.")
