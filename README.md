# AI Household Energy Consumption Predictor

Predict monthly household electricity consumption (kWh) for Indian households using a Random Forest Reressor trained on synthetic data (appliances, home size, city).

## Project structure
- train_model.py — generate dataset, train model, save artifacts  
- app.py — Tkinter GUI that loads the trained model for real-time predictions  
- requirements.txt — Python dependencies

## Setup

1. Create and activate a virtual environment
- Create:
```
python -m venv project_env
```
- Activate:
- Windows:
```
.\project_env\Scripts\activate
```
- macOS / Linux:
```
source project_env/bin/activate
```

2. Install dependencies
```
pip install -r requirements.txt
```

## Run

1. Train the model (run once or when retraining)
```
python train_model.py
```
This creates:
- energy_model.joblib — trained model  
- city_encoder.joblib — city preprocessing encoder  
- model_columns.txt — expected feature list

2. Start the GUI
```
python app.py
```
The GUI lets you enter household details and shows the predicted monthly consumption.
