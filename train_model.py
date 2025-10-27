# AI Energy Project - Step 1: Model Training
# This script builds a more complex model and saves it.
# Run this file ONCE to create the 'energy_model.joblib' file.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("--- AI Model Trainer ---")

# --- 1. Generate Expanded Synthetic Dataset ---
print("Generating synthetic dataset for 5000 households...")
np.random.seed(42)
num_households = 5000
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']

data = {
    'num_lights': np.random.randint(5, 25, num_households),
    'num_fans': np.random.randint(2, 10, num_households),
    'num_ac': np.random.randint(0, 5, num_households),
    'num_people': np.random.randint(1, 8, num_households),
    'house_size_sqft': np.random.randint(400, 3000, num_households),
    'appliance_age_years': np.random.randint(0, 10, num_households), # Age of major appliances
    'has_geyser': np.random.choice([0, 1], num_households, p=[0.3, 0.7]), # 1=Yes, 0=No
    'has_fridge': np.random.choice([0, 1], num_households, p=[0.1, 0.9]),
    'city': np.random.choice(cities, num_households)
}
df = pd.DataFrame(data)

# --- 2. Create the "True" Energy Consumption (More Complex Formula) ---
print("Calculating true energy consumption...")
WATTAGE = {
    'light': 12 / 1000,   # 12W LED
    'fan': 60 / 1000,    # 60W Ceiling Fan
    'ac': 1500 / 1000, # 1.5kW (1.5 Ton) AC
    'base': 0.5,         # Base daily kWh for TV, small items
    'geyser': 2.0,       # 2kW Geyser, 1.5 hours/day
    'fridge': 1.2,       # 1.2 kWh/day
    'people_proxy': 0.4  # Each person adds 0.4 kWh/day
}

HOURS = { 'light': 6, 'fan': 10, 'ac': 5 }

CITY_MULTIPLIER = {
    'Mumbai': 1.0, 'Delhi': 1.2, 'Bangalore': 0.8,
    'Chennai': 1.15, 'Kolkata': 1.1, 'Hyderabad': 1.05
}

def calculate_kwh(row):
    daily_kwh = WATTAGE['base']
    
    # Base appliances
    daily_kwh += row['num_lights'] * WATTAGE['light'] * HOURS['light']
    daily_kwh += row['num_fans'] * WATTAGE['fan'] * HOURS['fan']
    daily_kwh += row['num_ac'] * WATTAGE['ac'] * HOURS['ac']
    daily_kwh += row['num_people'] * WATTAGE['people_proxy']
    
    # Extra appliances
    daily_kwh += row['has_geyser'] * WATTAGE['geyser'] * 1.5 # 1.5 hours/day
    daily_kwh += row['has_fridge'] * WATTAGE['fridge']
    
    # House size factor
    daily_kwh += (row['house_size_sqft'] / 1000) * 0.5 # 0.5 kWh per 1000 sqft
    
    # Appliance age inefficiency factor (1% less efficient per year)
    inefficiency_factor = 1 + (row['appliance_age_years'] * 0.01)
    
    # Apply factors
    noise = np.random.uniform(0.95, 1.05) # 5% randomness
    monthly_kwh = daily_kwh * 30 * CITY_MULTIPLIER[row['city']] * inefficiency_factor * noise
    return monthly_kwh

df['monthly_kwh'] = df.apply(calculate_kwh, axis=1)

# --- 3. Preprocess Data for ML ---
print("Preprocessing data (One-Hot Encoding)...")
X = df.drop('monthly_kwh', axis=1)
y = df['monthly_kwh']

categorical_features = ['city']
numeric_features = [
    'num_lights', 'num_fans', 'num_ac', 'num_people', 
    'house_size_sqft', 'appliance_age_years', 'has_geyser', 'has_fridge'
]

# Create and fit the encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(X[categorical_features])
encoded_cols = encoder.get_feature_names_out(categorical_features)
X_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=X.index)

# Combine all features
X_processed = pd.concat([X[numeric_features], X_encoded], axis=1)

# --- 4. Split Data and Train Model ---
print("Splitting data and training RandomForest model...")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all cores
model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R-squared (R2) Score on Test Data: {r2:.4f}")

# --- 6. Save the Model and Encoder ---
print("\nSaving model, encoder, and column list...")
joblib.dump(model, 'energy_model.joblib')
joblib.dump(encoder, 'city_encoder.joblib')

# Save column names in the correct order for the GUI
with open('model_columns.txt', 'w') as f:
    for col in X_processed.columns:
        f.write(col + '\n')

print("--- Training complete. Model saved as 'energy_model.joblib' ---")

