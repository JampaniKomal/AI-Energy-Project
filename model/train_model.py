# AI Energy Project - Step 1 (V2): Enhanced Model Training
# This script builds a more complex model with Gradient Boosting
# and saves the model and preprocessor.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

print("--- AI Model Trainer V2 ---")

# Define the output directory relative to this script
# os.path.dirname(__file__) gets the directory of the current script (model/)
# os.path.join(..., '..') goes up one level (to the project root)
# os.path.join(..., 'saved_models') points to the saved_models folder
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, '..', 'saved_models')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- 1. Generate Expanded Synthetic Dataset ---
print("Generating enhanced synthetic dataset for 5000 households...")
np.random.seed(42)
num_households = 5000
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
building_types = ['Apartment', 'Independent House', 'Villa']

data = {
    'num_lights': np.random.randint(5, 25, num_households),
    'num_fans': np.random.randint(2, 10, num_households),
    'num_ac': np.random.randint(0, 5, num_households),
    'num_people': np.random.randint(1, 8, num_households),
    'house_size_sqft': np.random.randint(400, 3500, num_households),
    'appliance_age_years': np.random.randint(0, 12, num_households),
    'has_geyser': np.random.choice([0, 1], num_households, p=[0.3, 0.7]),
    'has_fridge': np.random.choice([0, 1], num_households, p=[0.1, 0.9]),
    'city': np.random.choice(cities, num_households),
    'building_type': np.random.choice(building_types, num_households, p=[0.6, 0.3, 0.1]),
    'is_metro': np.random.choice([0, 1], num_households, p=[0.2, 0.8]) # 1 if in a major metro area
}
df = pd.DataFrame(data)

# --- 2. Create the "True" Energy Consumption (More Complex Formula) ---
print("Calculating true energy consumption...")
WATTAGE = {
    'light': 12 / 1000, 'fan': 60 / 1000, 'ac': 1500 / 1000,
    'base': 0.5, 'geyser': 2.0, 'fridge': 1.2, 'people_proxy': 0.45
}
HOURS = { 'light': 6.5, 'fan': 11, 'ac': 5.5 }
CITY_MULTIPLIER = {
    'Mumbai': 1.0, 'Delhi': 1.25, 'Bangalore': 0.8,
    'Chennai': 1.20, 'Kolkata': 1.1, 'Hyderabad': 1.05
}
BUILDING_FACTOR = {'Apartment': 1.0, 'Independent House': 1.1, 'Villa': 1.3}
METRO_FACTOR = {0: 1.05, 1: 1.0} # Non-metro slightly higher due to grid variability assumption

def calculate_kwh(row):
    daily_kwh = WATTAGE['base']
    daily_kwh += row['num_lights'] * WATTAGE['light'] * HOURS['light']
    daily_kwh += row['num_fans'] * WATTAGE['fan'] * HOURS['fan']
    daily_kwh += row['num_ac'] * WATTAGE['ac'] * HOURS['ac']
    daily_kwh += row['num_people'] * WATTAGE['people_proxy']
    daily_kwh += row['has_geyser'] * WATTAGE['geyser'] * 1.5
    daily_kwh += row['has_fridge'] * WATTAGE['fridge']
    daily_kwh += (row['house_size_sqft'] / 1000) * 0.6 # Increased impact

    inefficiency_factor = 1 + (row['appliance_age_years'] * 0.015) # Increased impact

    noise = np.random.uniform(0.93, 1.07) # Increased randomness

    monthly_kwh = (daily_kwh * 30 *
                   CITY_MULTIPLIER[row['city']] *
                   BUILDING_FACTOR[row['building_type']] *
                   METRO_FACTOR[row['is_metro']] *
                   inefficiency_factor * noise)
    return max(50, monthly_kwh) # Ensure minimum consumption

df['monthly_kwh'] = df.apply(calculate_kwh, axis=1)

# --- 3. Define Preprocessing ---
print("Defining preprocessing steps...")
X = df.drop('monthly_kwh', axis=1)
y = df['monthly_kwh']

# Identify feature types
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Create preprocessing pipelines for numeric and categorical features
# Numeric features will be scaled
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical features will be one-hot encoded
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any columns not specified (none in this case)
)

# --- 4. Define the Model Pipeline ---
# Chain the preprocessor and the Gradient Boosting Regressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# --- 5. Split Data and Train Model ---
print("Splitting data and training Gradient Boosting model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the entire pipeline
model_pipeline.fit(X_train, y_train)

# --- 6. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)

# --- WORKAROUND for 'squared' argument error ---
# Calculate RMSE manually: sqrt(MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# Original line: rmse = mean_squared_error(y_test, y_pred, squared=False)
# --- End Workaround ---

print(f"Model R-squared (R2) Score on Test Data: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse:.2f} kWh")

# --- 7. Save the Model Pipeline and Feature Names ---
# We use os.path.abspath to get a clear, full path for debugging
save_path = os.path.abspath(output_dir)
print(f"\nAbsolute path being used for saving: {save_path}")
print(f"Saving model pipeline and feature names to '{output_dir}'...")

joblib.dump(model_pipeline, os.path.join(output_dir, 'energy_model.joblib'))

# Get feature names after transformation (important for the GUI)
try:
    feature_names = numeric_features + \
                    list(model_pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(categorical_features))
except Exception as e:
    print(f"Could not get feature names from pipeline, falling back. Error: {e}")
    # Fallback for older sklearn versions
    ohe_features = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_features)


with open(os.path.join(output_dir, 'model_columns.txt'), 'w') as f:
    for name in feature_names:
        f.write(name + '\n')

# Save the original categorical feature names needed for the GUI dropdowns
original_categorical_features = {
    'city': cities,
    'building_type': building_types,
}
joblib.dump(original_categorical_features, os.path.join(output_dir, 'categorical_options.joblib'))


print("--- Training complete. Model artefacts saved. ---")

