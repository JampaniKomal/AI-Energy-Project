# AI Energy Project - Step 2 (V2): Upgraded GUI
# This version uses customtkinter for a modern look
# and provides more analysis (cost and breakdown).

import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
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
        
    CITY_OPTIONS = encoder.categories_[0]

except FileNotFoundError:
    messagebox.showerror(
        "Error", 
        "Model files not found! Please run 'train_model.py' first."
    )
    exit()
except Exception as e:
    messagebox.showerror("Error", f"An error occurred loading model files: {e}")
    exit()
    
# --- 2. Define Constants for Breakdown ---
# These are the *approximate* rules from train_model.py
# We use them to give the user an *estimated* breakdown.
WATTAGE = {
    'light': 12 / 1000,   'fan': 60 / 1000,    'ac': 1500 / 1000,
    'base': 0.5, 'geyser': 2.0, 'fridge': 1.2, 'people_proxy': 0.4
}
HOURS = { 'light': 6, 'fan': 10, 'ac': 5 }
CITY_MULTIPLIER = {
    'Mumbai': 1.0, 'Delhi': 1.2, 'Bangalore': 0.8,
    'Chennai': 1.15, 'Kolkata': 1.1, 'Hyderabad': 1.05
}

# --- 3. Create the Prediction Function ---
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
        price_per_kwh = float(entry_price.get())

        # --- b. Prepare data for the model (same as before) ---
        input_df = pd.DataFrame([input_data])
        numeric_features = [
            'num_lights', 'num_fans', 'num_ac', 'num_people', 
            'house_size_sqft', 'appliance_age_years', 'has_geyser', 'has_fridge'
        ]
        categorical_features = ['city']
        encoded_cats = encoder.transform(input_df[categorical_features])
        encoded_cols = encoder.get_feature_names_out(categorical_features)
        df_numeric = pd.DataFrame(input_df[numeric_features])
        df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols)
        df_processed = pd.concat([df_numeric, df_encoded], axis=1)
        df_final = df_processed.reindex(columns=model_columns)

        # --- c. Make AI Prediction ---
        predicted_kwh = model.predict(df_final)[0]
        predicted_cost = predicted_kwh * price_per_kwh

        # --- d. Calculate Estimated Breakdown (for analysis) ---
        daily_kwh_light = input_data['num_lights'] * WATTAGE['light'] * HOURS['light']
        daily_kwh_fan = input_data['num_fans'] * WATTAGE['fan'] * HOURS['fan']
        daily_kwh_ac = input_data['num_ac'] * WATTAGE['ac'] * HOURS['ac']
        daily_kwh_geyser = input_data['has_geyser'] * WATTAGE['geyser'] * 1.5
        daily_kwh_fridge = input_data['has_fridge'] * WATTAGE['fridge']
        daily_kwh_other = WATTAGE['base'] + (input_data['num_people'] * WATTAGE['people_proxy']) + (input_data['house_size_sqft'] / 1000) * 0.5
        
        # Apply multipliers
        multiplier = CITY_MULTIPLIER[input_data['city']] * (1 + (input_data['appliance_age_years'] * 0.01))
        
        # Get total daily kWh from breakdown
        total_daily_kwh_breakdown = (
            daily_kwh_light + daily_kwh_fan + daily_kwh_ac + 
            daily_kwh_geyser + daily_kwh_fridge + daily_kwh_other
        ) * multiplier
        
        # Calculate percentages
        breakdown_text = (
            f"Lights: {(daily_kwh_light * multiplier * 30 * price_per_kwh):.0f} Rs\n"
            f"Fans: {(daily_kwh_fan * multiplier * 30 * price_per_kwh):.0f} Rs\n"
            f"ACs: {(daily_kwh_ac * multiplier * 30 * price_per_kwh):.0f} Rs\n"
            f"Geyser: {(daily_kwh_geyser * multiplier * 30 * price_per_kwh):.0f} Rs\n"
            f"Fridge: {(daily_kwh_fridge * multiplier * 30 * price_per_kwh):.0f} Rs\n"
            f"Other: {(daily_kwh_other * multiplier * 30 * price_per_kwh):.0f} Rs"
        )
        
        # --- e. Display Results ---
        result_total_label.configure(
            text=f"Total Est. Bill: {predicted_cost:.2f} Rs",
            text_color="#00A9FF"
        )
        result_kwh_label.configure(
            text=f"(Based on AI Model prediction of {predicted_kwh:.2f} kWh)"
        )
        result_breakdown_header.configure(text="Estimated Cost Breakdown:")
        result_breakdown_label.configure(text=breakdown_text, justify=tk.LEFT)

    except ValueError:
        messagebox.showerror("Input Error", "Please ensure all fields have valid numbers.")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred: {e}")

# --- 4. Set up the GUI ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("AI Energy Consumption Predictor")
root.geometry("450x700")

# Main frame
main_frame = ctk.CTkFrame(root)
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

title_label = ctk.CTkLabel(main_frame, text="Household Energy Predictor", font=("Helvetica", 24, "bold"))
title_label.pack(pady=(0, 20))

# --- Input Fields Frame ---
fields_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
fields_frame.pack(fill="x", pady=5)

# Grid layout for inputs
fields_frame.grid_columnconfigure(0, weight=1)
fields_frame.grid_columnconfigure(1, weight=2)

def create_entry(row, label_text, default_value):
    label = ctk.CTkLabel(fields_frame, text=label_text)
    label.grid(row=row, column=0, sticky="w", padx=10, pady=8)
    entry = ctk.CTkEntry(fields_frame)
    entry.grid(row=row, column=1, sticky="ew", padx=10, pady=8)
    entry.insert(0, default_value)
    return entry

entry_lights = create_entry(0, "Number of Lights:", "12")
entry_fans = create_entry(1, "Number of Fans:", "4")
entry_ac = create_entry(2, "Number of ACs:", "2")
entry_people = create_entry(3, "Number of People:", "3")
entry_sqft = create_entry(4, "House Size (sq. ft.):", "1200")
entry_age = create_entry(5, "Appliance Age (years):", "3")
entry_price = create_entry(6, "Price per kWh (Rs):", "8.0") # New field

# Checkbox Inputs
var_geyser = ctk.IntVar(value=1)
chk_geyser = ctk.CTkCheckBox(fields_frame, text="Has Geyser (Water Heater)", variable=var_geyser)
chk_geyser.grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=8)

var_fridge = ctk.IntVar(value=1)
chk_fridge = ctk.CTkCheckBox(fields_frame, text="Has Refrigerator", variable=var_fridge)
chk_fridge.grid(row=8, column=0, columnspan=2, sticky="w", padx=10, pady=8)

# Dropdown Input
label_city = ctk.CTkLabel(fields_frame, text="City:")
label_city.grid(row=9, column=0, sticky="w", padx=10, pady=8)
var_city = ctk.StringVar(value=CITY_OPTIONS[1]) # Default to 'Delhi'
city_menu = ctk.CTkOptionMenu(fields_frame, variable=var_city, values=list(CITY_OPTIONS))
city_menu.grid(row=9, column=1, sticky="ew", padx=10, pady=8)

# --- Predict Button ---
predict_button = ctk.CTkButton(main_frame, text="Predict Consumption", command=predict_energy, font=("Helvetica", 16, "bold"), height=40)
predict_button.pack(pady=20, fill="x", padx=10)

# --- Result Frame ---
result_frame = ctk.CTkFrame(main_frame, fg_color="gray20")
result_frame.pack(fill="x", padx=10)

result_total_label = ctk.CTkLabel(result_frame, text="", font=("Helvetica", 22, "bold"))
result_total_label.pack(pady=(10, 0))

result_kwh_label = ctk.CTkLabel(result_frame, text="Click 'Predict' to see the result", font=("Helvetica", 12), text_color="gray")
result_kwh_label.pack()

result_breakdown_header = ctk.CTkLabel(result_frame, text="", font=("Helvetica", 14, "bold", "underline"))
result_breakdown_header.pack(pady=(15, 5))

result_breakdown_label = ctk.CTkLabel(result_frame, text="", font=("Monospace", 12))
result_breakdown_label.pack(pady=(0, 15), anchor="w", padx=20)

root.mainloop()

