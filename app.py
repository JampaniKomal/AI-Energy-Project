# AI Energy Project - Step 2: GUI Application
# This script loads the trained model and provides a GUI
# to make predictions.

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. Load Model, Encoder, and Columns ---
try:
    model = joblib.load('energy_model.joblib')
    encoder = joblib.load('city_encoder.joblib')
    
    with open('model_columns.txt', 'r') as f:
        model_columns = [line.strip() for line in f.readlines()]
        
    # Get the city names from the encoder
    CITY_OPTIONS = encoder.categories_[0]

except FileNotFoundError:
    messagebox.showerror(
        "Error", 
        "Model files not found!\n\n"
        "Please run 'train_model.py' first to generate:\n"
        "- energy_model.joblib\n"
        "- city_encoder.joblib\n"
        "- model_columns.txt"
    )
    exit()
except Exception as e:
    messagebox.showerror("Error", f"An error occurred loading model files: {e}")
    exit()

# --- 2. Create the Prediction Function ---
def predict_energy():
    try:
        # --- a. Collect input data from GUI ---
        input_data = {
            'num_lights': int(entry_lights.get()),
            'num_fans': int(entry_fans.get()),
            'num_ac': int(entry_ac.get()),
            'num_people': int(entry_people.get()),
            'house_size_sqft': float(entry_sqft.get()),
            'appliance_age_years': int(entry_age.get()),
            'has_geyser': int(var_geyser.get()),
            'has_fridge': int(var_fridge.get()),
            'city': var_city.get()
        }

        # --- b. Prepare data for the model ---
        # Create a DataFrame from the single input
        input_df = pd.DataFrame([input_data])
        
        # Separate numeric and categorical
        numeric_features = [
            'num_lights', 'num_fans', 'num_ac', 'num_people', 
            'house_size_sqft', 'appliance_age_years', 'has_geyser', 'has_fridge'
        ]
        categorical_features = ['city']

        # Apply the loaded One-Hot Encoder
        encoded_cats = encoder.transform(input_df[categorical_features])
        encoded_cols = encoder.get_feature_names_out(categorical_features)
        
        # Create DataFrames
        df_numeric = pd.DataFrame(input_df[numeric_features])
        df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols)
        
        # Combine numeric and encoded features
        df_processed = pd.concat([df_numeric, df_encoded], axis=1)
        
        # --- c. Ensure column order matches training ---
        # Reorder the DataFrame to match the exact order the model was trained on
        df_final = df_processed.reindex(columns=model_columns)

        # --- d. Make prediction ---
        prediction = model.predict(df_final)
        
        # --- e. Display result ---
        result_label.config(
            text=f"Predicted Monthly Consumption:\n{prediction[0]:.2f} kWh",
            font=("Helvetica", 14, "bold"),
            foreground="blue"
        )

    except ValueError:
        messagebox.showerror("Input Error", "Please ensure all numeric fields are filled with valid numbers.")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred: {e}")

# --- 3. Set up the GUI ---
root = tk.Tk()
root.title("AI Energy Consumption Predictor")
root.geometry("450x550")

# Use a themed style
style = ttk.Style(root)
style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

# Main frame
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(expand=True, fill=tk.BOTH)

# Title
title_label = ttk.Label(main_frame, text="Household Energy Predictor", font=("Helvetica", 18, "bold"))
title_label.pack(pady=(0, 20))

# --- Input Fields ---
# Using a grid layout for alignment
fields_frame = ttk.Frame(main_frame)
fields_frame.pack(fill=tk.X, pady=10)

# Configure grid columns to have equal weight for centering
fields_frame.grid_columnconfigure(0, weight=1)
fields_frame.grid_columnconfigure(1, weight=1)

# Numeric Inputs
ttk.Label(fields_frame, text="Number of Lights:").grid(row=0, column=0, sticky=tk.W, pady=5)
entry_lights = ttk.Entry(fields_frame, width=15)
entry_lights.grid(row=0, column=1, sticky=tk.W, pady=5)
entry_lights.insert(0, "12") # Default value

ttk.Label(fields_frame, text="Number of Fans:").grid(row=1, column=0, sticky=tk.W, pady=5)
entry_fans = ttk.Entry(fields_frame, width=15)
entry_fans.grid(row=1, column=1, sticky=tk.W, pady=5)
entry_fans.insert(0, "4")

ttk.Label(fields_frame, text="Number of ACs:").grid(row=2, column=0, sticky=tk.W, pady=5)
entry_ac = ttk.Entry(fields_frame, width=15)
entry_ac.grid(row=2, column=1, sticky=tk.W, pady=5)
entry_ac.insert(0, "2")

ttk.Label(fields_frame, text="Number of People:").grid(row=3, column=0, sticky=tk.W, pady=5)
entry_people = ttk.Entry(fields_frame, width=15)
entry_people.grid(row=3, column=1, sticky=tk.W, pady=5)
entry_people.insert(0, "3")

ttk.Label(fields_frame, text="House Size (sq. ft.):").grid(row=4, column=0, sticky=tk.W, pady=5)
entry_sqft = ttk.Entry(fields_frame, width=15)
entry_sqft.grid(row=4, column=1, sticky=tk.W, pady=5)
entry_sqft.insert(0, "1200")

ttk.Label(fields_frame, text="Appliance Age (years):").grid(row=5, column=0, sticky=tk.W, pady=5)
entry_age = ttk.Entry(fields_frame, width=15)
entry_age.grid(row=5, column=1, sticky=tk.W, pady=5)
entry_age.insert(0, "3")

# Checkbox Inputs
var_geyser = tk.IntVar(value=1)
chk_geyser = ttk.Checkbutton(fields_frame, text="Has Geyser (Water Heater)", variable=var_geyser)
chk_geyser.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)

var_fridge = tk.IntVar(value=1)
chk_fridge = ttk.Checkbutton(fields_frame, text="Has Refrigerator", variable=var_fridge)
chk_fridge.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)

# Dropdown Input
ttk.Label(fields_frame, text="City:").grid(row=8, column=0, sticky=tk.W, pady=5)
var_city = tk.StringVar(value=CITY_OPTIONS[1]) # Default to 'Delhi'
city_menu = ttk.OptionMenu(fields_frame, var_city, CITY_OPTIONS[0], *CITY_OPTIONS)
city_menu.grid(row=8, column=1, sticky=tk.W, pady=5)

# --- Predict Button ---
predict_button = ttk.Button(main_frame, text="Predict Consumption", command=predict_energy, style='Accent.TButton')
style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'), padding=10)
predict_button.pack(pady=20)

# --- Result Label ---
result_label = ttk.Label(main_frame, text="Click 'Predict' to see the result", font=("Helvetica", 12))
result_label.pack(pady=10)

# --- Run the GUI ---
root.mainloop()

