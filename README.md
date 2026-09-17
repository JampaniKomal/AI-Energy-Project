# AI Household Energy Consumption Predictor

Predicts monthly household electricity consumption (kWh) for Indian
households from a Gradient Boosting model, served through a desktop GUI
that also shows an estimated cost breakdown per appliance category.

## Why this exists

A small end-to-end ML project: generate a realistic synthetic dataset,
train a proper scikit-learn pipeline on it, and serve real-time
predictions through a usable desktop app rather than a notebook.

## Features

- Gradient Boosting Regressor wrapped in a full scikit-learn `Pipeline`
  (`ColumnTransformer` with `StandardScaler` for numeric features and
  `OneHotEncoder` for categorical ones), so preprocessing and the model
  are saved and loaded as a single artifact.
- Synthetic data generator modeling realistic Indian-household factors:
  per-city cost multipliers, building type (apartment/independent
  house/villa), metro-area effects, and appliance-age inefficiency.
- `customtkinter` desktop GUI for entering household details and getting
  a prediction, plus an estimated Rs. cost breakdown by appliance category.

## Project structure

- `model/train_model.py` — generates the synthetic dataset, trains the
  pipeline, and saves it to `saved_models/`. Run this once (or whenever
  retraining).
- `app/app.py` — the GUI. Loads the trained pipeline from `saved_models/`
  and predicts on whatever household details you enter.
- `tests/` — pytest suite for the underlying consumption formula.

## How it works

1. `model/train_model.py` builds a 5,000-household synthetic dataset from
   a hand-written "true" consumption formula (appliance counts, house
   size, city/building/metro factors, appliance age, and some random
   noise), trains the pipeline on it, and saves three files to
   `saved_models/`: `energy_model.joblib` (the fitted pipeline),
   `model_columns.txt` (the transformed feature names), and
   `categorical_options.joblib` (the dropdown options for the GUI).
2. `app/app.py` loads those three files and, per prediction, builds a
   single-row DataFrame from the form inputs and calls
   `model_pipeline.predict(...)`.
3. The on-screen cost breakdown by appliance (lights/fans/AC/geyser/
   fridge/other) is a separate, simpler heuristic estimate computed
   directly in the GUI — it is not something the trained model itself
   outputs. See Known Limitations.

## Installation

```
python -m venv project_env
```
Activate it:
- Windows: `.\project_env\Scripts\activate`
- macOS/Linux: `source project_env/bin/activate`

```
pip install -r requirements.txt
```

## Usage

```
python model/train_model.py   # trains and saves the model (run once)
python app/app.py             # launches the GUI
```

## Testing

```
pip install -r requirements-dev.txt
pytest
```

16 tests cover the core `calculate_kwh` consumption formula: the minimum
floor, the direction of every multiplier (city, building type, metro,
appliance age), and that more appliances always predict higher
consumption. Verified by actually training the model end-to-end: on a
held-out test split it reaches R² = 0.9868 and RMSE = 57.21 kWh at
recovering the synthetic formula.

## Known Limitations

- Trained entirely on synthetic data generated from a known formula, not
  real metered household data. The R²/RMSE above measure how well the
  model recovers that synthetic formula, not real-world predictive
  accuracy.
- The per-appliance cost breakdown shown in the GUI is a separate,
  hand-written heuristic estimate, not an output of the trained model —
  the model only predicts a single total kWh figure. The two use
  similar-looking constants but are computed independently, so the
  breakdown should be read as illustrative, not as the model's own
  reasoning.
- No input validation range-checking in the GUI beyond "is this a valid
  number" — e.g. a household of 500 people would still be accepted.

## License

MIT — see [LICENSE](LICENSE).
