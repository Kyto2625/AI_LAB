"""
app.py
---------------------------------------------------------------------------
Petroleum Refinery Yield Prediction System - Flask Web Application

Run with:
    python app.py
Then open the link shown in the terminal (usually http://127.0.0.1:5000)

NOTE: You must run train_models.py at least once before starting the app,
so that model/best_model.pkl exists.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained model + metadata once at startup
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
target_columns = joblib.load(os.path.join(MODEL_DIR, "target_columns.pkl"))
best_model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))

try:
    results_df = pd.read_csv(os.path.join(MODEL_DIR, "model_comparison_results.csv"), index_col=0)
except FileNotFoundError:
    results_df = None

GEOLOGICAL_FORMATIONS = ["Sandstone", "Carbonate", "Shale", "Limestone", "Unconsolidated"]
CRUDE_SOURCES = ["Gulf Coast", "Permian Basin", "Mid-Continent", "Rocky Mountain",
                 "California", "Foreign-Import"]

FRACTION_LABELS = {
    "light_gas_yield_pct": "Light Gas",
    "kerosene_yield_pct": "Kerosene",
    "gas_oil_yield_pct": "Gas Oil",
    "lubricant_yield_pct": "Lubricant Fractions",
    "residual_yield_pct": "Residual Fraction",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "input_form.html",
            formations=GEOLOGICAL_FORMATIONS,
            sources=CRUDE_SOURCES,
        )

    # ---- Handle form submission (POST) ----
    try:
        input_data = pd.DataFrame([{
            "api_gravity": float(request.form["api_gravity"]),
            "specific_gravity": float(request.form["specific_gravity"]),
            "pour_point_f": float(request.form["pour_point_f"]),
            "viscosity_cst": float(request.form["viscosity_cst"]),
            "sulfur_pct": float(request.form["sulfur_pct"]),
            "nitrogen_pct": float(request.form["nitrogen_pct"]),
            "geological_formation": request.form["geological_formation"],
            "crude_source": request.form["crude_source"],
        }])
    except (KeyError, ValueError) as e:
        return f"Invalid input: {e}", 400

    prediction = model.predict(input_data)[0]
    # Normalize to sum to 100% for a clean display (model output may drift slightly)
    prediction = np.clip(prediction, 0, None)
    prediction = prediction / prediction.sum() * 100

    results = [
        {"fraction": FRACTION_LABELS[col], "value": round(float(val), 2)}
        for col, val in zip(target_columns, prediction)
    ]

    return render_template(
        "result.html",
        results=results,
        model_name=best_model_name,
        input_summary=input_data.iloc[0].to_dict(),
    )


@app.route("/model-info")
def model_info():
    comparison = None
    if results_df is not None:
        comparison = results_df.round(3).to_dict(orient="index")
    return render_template(
        "model_info.html",
        comparison=comparison,
        best_model_name=best_model_name,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)