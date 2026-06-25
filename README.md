# Petroleum Refinery Yield Prediction System

A web-based machine learning application that predicts petroleum refinery
fractional yields (light gas, kerosene, gas oil, lubricant fractions,
residual fraction) from crude oil physicochemical properties, comparing
**Random Forest Regressor**, **XGBoost Regressor**, and **Support Vector
Regressor (SVR)**.

## ⚠️ About the dataset (read this first)

This project is designed around the **OpenEI Crude Oil Analysis (COA)
Database** (https://data.openei.org/submissions/178). However, the real
file is only distributed as a 1990s Microsoft Access database (`coadb.mdb`)
inside a `.zip` — not a clean CSV — so it can't be auto-loaded by code.

To make the project **fully runnable right now**, `data/generate_synthetic_data.py`
creates a placeholder dataset with the **exact same column schema**, built
using real petroleum-engineering relationships (lighter crude → more light
fractions, heavier/more viscous crude → more residue, etc.).

**Before your final submission**, swap in the real data:
1. Download: https://data.openei.org/files/178/coa.zip
2. Open `coadb.mdb` in Microsoft Access (or the bundled Excel file).
3. Export the main crude assay table to CSV, matching these column names
   (rename columns as needed):
   `api_gravity, specific_gravity, pour_point_f, viscosity_cst, sulfur_pct,
   nitrogen_pct, geological_formation, crude_source, light_gas_yield_pct,
   kerosene_yield_pct, gas_oil_yield_pct, lubricant_yield_pct, residual_yield_pct`
4. Save it as `data/crude_oil_data.csv` (overwrite the placeholder).
5. Re-run `train_models.py`. No other code changes are needed.

## Project structure

```
petro_yield_project/
├── data/
│   ├── generate_synthetic_data.py   # creates placeholder dataset
│   └── crude_oil_data.csv           # generated dataset (input to training)
├── model/                           # saved model + metrics (created by training)
├── static/                          # CSS + generated chart images
├── templates/                       # HTML pages (Flask/Jinja2)
├── train_models.py                  # trains & compares the 3 regressors
├── app.py                           # Flask web application
└── requirements.txt
```

## How to run (PyCharm or terminal)

1. **Create a virtual environment** (PyCharm: New Project will offer this automatically).
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Generate the dataset** (only needed once, or whenever you swap in real data):
   ```
   python data/generate_synthetic_data.py
   ```
4. **Train and compare the models:**
   ```
   python train_models.py
   ```
   This prints MAE / RMSE / R² for all three algorithms, saves the
   best-performing model to `model/best_model.pkl`, and saves two charts
   to `static/` (model comparison + feature importance).
5. **Run the web app:**
   ```
   python app.py
   ```
   Open the link shown in the terminal (usually `http://127.0.0.1:5000`).

## Web app pages

| Route | Page |
|---|---|
| `/` | Homepage — project overview |
| `/predict` | Input form → prediction results |
| `/model-info` | Model comparison table + charts |

## Notes

- All three models are wrapped in `MultiOutputRegressor` since the system
  predicts **5 yield fractions simultaneously** from one set of inputs.
- The best model (by R² Score) is automatically selected and deployed —
  no manual step required.
- Predicted yields are renormalized to sum to exactly 100% for clean display,
  since regression outputs may drift slightly off due to independent
  per-fraction prediction.
