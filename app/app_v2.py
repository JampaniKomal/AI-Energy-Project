# AI Energy Project - Step 2 (V2.1): GUI with Enhanced Model Loading
# Uses customtkinter and loads the pipeline from train_model_v2.py

import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import joblib
import pandas as pd
import numpy as np
import os

# --- Constants ---
MODEL_DIR = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models')))

# --- 1. Load Model Pipeline, Columns, and Categorical Options ---
try:
    model_pipeline = joblib.load(os.path.join(MODEL_DIR, 'energy_model.joblib'))
    
    with open(os.path.join(MODEL_DIR, 'model_columns.txt'), 'r') as f:
        # These are the *transformed* column names the pipeline expects
        transformed_feature_names = [line.strip() for line in f.readlines()]
        
    # Load the original options for dropdowns
    categorical_options = joblib.load(os.path.join(MODEL_DIR, 'categorical_options.joblib'))
    CITY_OPTIONS = categorical_options['city']
    BUILDING_TYPE_OPTIONS = categorical_options['building_type']

except FileNotFoundError:
    messagebox.showerror(
        "Error", 
        f"Model files not found in '{MODEL_DIR}'!\n\n"
        "Please run 'model/train_model_v2.py' first."
    )
    exit()
except Exception as e:
    messagebox.showerror("Error", f"An error occurred loading model files: {e}")
    exit()
    
# --- 2. Define Constants for Breakdown (same as train_model_v2) ---
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
METRO_FACTOR = {0: 1.05, 1: 1.0}

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
            'is_metro': int(var_metro.get()), # New input
            'city': var_city.get(),
            'building_type': var_building.get() # New input
        }
        price_per_kwh = float(entry_price.get())

        # --- b. Prepare data using the loaded pipeline ---
        # Create a DataFrame from the single input
        # Important: Column names MUST match those used during training
        input_df = pd.DataFrame([input_data])

        # --- c. Make AI Prediction ---
        # The pipeline handles preprocessing AND prediction
        predicted_kwh = model_pipeline.predict(input_df)[0]
        predicted_cost = predicted_kwh * price_per_kwh

        # --- d. Calculate Estimated Breakdown (for analysis - updated) ---
        daily_kwh_light = input_data['num_lights'] * WATTAGE['light'] * HOURS['light']
        daily_kwh_fan = input_data['num_fans'] * WATTAGE['fan'] * HOURS['fan']
        daily_kwh_ac = input_data['num_ac'] * WATTAGE['ac'] * HOURS['ac']
        daily_kwh_geyser = input_data['has_geyser'] * WATTAGE['geyser'] * 1.5
        daily_kwh_fridge = input_data['has_fridge'] * WATTAGE['fridge']
        daily_kwh_other = WATTAGE['base'] + (input_data['num_people'] * WATTAGE['people_proxy']) + (input_data['house_size_sqft'] / 1000) * 0.6
        
        multiplier = (CITY_MULTIPLIER[input_data['city']] *
                      BUILDING_FACTOR[input_data['building_type']] *
                      METRO_FACTOR[input_data['is_metro']] *
                      (1 + (input_data['appliance_age_years'] * 0.015)))
        
        # Calculate monthly costs for breakdown
        cost_light = daily_kwh_light * multiplier * 30 * price_per_kwh
        cost_fan = daily_kwh_fan * multiplier * 30 * price_per_kwh
        cost_ac = daily_kwh_ac * multiplier * 30 * price_per_kwh
        cost_geyser = daily_kwh_geyser * multiplier * 30 * price_per_kwh
        cost_fridge = daily_kwh_fridge * multiplier * 30 * price_per_kwh
        cost_other = daily_kwh_other * multiplier * 30 * price_per_kwh

        breakdown_text = (
            f"Lights: {cost_light:.0f} Rs\n"
            f"Fans:   {cost_fan:.0f} Rs\n"
            f"ACs:    {cost_ac:.0f} Rs\n"
            f"Geyser: {cost_geyser:.0f} Rs\n"
            f"Fridge: {cost_fridge:.0f} Rs\n"
            f"Other:  {cost_other:.0f} Rs"
        )
        
        # --- e. Display Results ---
        result_total_label.configure(
            text=f"Total Est. Bill: {predicted_cost:.2f} Rs",
            text_color="#33FF57" # Brighter Green
        )
        result_kwh_label.configure(
            text=f"(AI Model Prediction: {predicted_kwh:.2f} kWh)"
        )
        result_breakdown_header.configure(text="Estimated Cost Breakdown:")
        result_breakdown_label.configure(text=breakdown_text, justify=tk.LEFT)

    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}\nPlease ensure all fields have valid numbers.")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred: {e}")

# --- 4. Set up the GUI ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green") # Changed theme

root = ctk.CTk()
root.title("AI Energy Consumption Predictor V2")
root.geometry("480x780") # Increased size

# Main frame
main_frame = ctk.CTkFrame(root)
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

title_label = ctk.CTkLabel(main_frame, text="Household Energy Predictor", font=ctk.CTkFont(size=24, weight="bold"))
title_label.pack(pady=(0, 20))

# --- Input Fields Frame ---
fields_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
fields_frame.pack(fill="x", pady=5)
fields_frame.grid_columnconfigure(0, weight=1)
fields_frame.grid_columnconfigure(1, weight=2)

def create_entry(row, label_text, default_value):
    label = ctk.CTkLabel(fields_frame, text=label_text)
    label.grid(row=row, column=0, sticky="w", padx=10, pady=8)
    entry = ctk.CTkEntry(fields_frame, width=150) # Increased width
    entry.grid(row=row, column=1, sticky="ew", padx=10, pady=8)
    entry.insert(0, default_value)
    return entry

entry_lights = create_entry(0, "Number of Lights:", "12")
entry_fans = create_entry(1, "Number of Fans:", "4")
entry_ac = create_entry(2, "Number of ACs:", "2")
entry_people = create_entry(3, "Number of People:", "3")
entry_sqft = create_entry(4, "House Size (sq. ft.):", "1200")
entry_age = create_entry(5, "Appliance Age (years):", "3")
entry_price = create_entry(6, "Price per kWh (Rs):", "8.0")

# Checkbox Inputs (row 7, 8, 9)
var_geyser = ctk.IntVar(value=1)
chk_geyser = ctk.CTkCheckBox(fields_frame, text="Has Geyser (Water Heater)", variable=var_geyser)
chk_geyser.grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=8)

var_fridge = ctk.IntVar(value=1)
chk_fridge = ctk.CTkCheckBox(fields_frame, text="Has Refrigerator", variable=var_fridge)
chk_fridge.grid(row=8, column=0, columnspan=2, sticky="w", padx=10, pady=8)

var_metro = ctk.IntVar(value=1)
chk_metro = ctk.CTkCheckBox(fields_frame, text="Located in Metro City Area", variable=var_metro)
chk_metro.grid(row=9, column=0, columnspan=2, sticky="w", padx=10, pady=8)

# Dropdown Inputs (row 10, 11)
label_city = ctk.CTkLabel(fields_frame, text="City:")
label_city.grid(row=10, column=0, sticky="w", padx=10, pady=8)
var_city = ctk.StringVar(value=CITY_OPTIONS[1]) # Default to 'Delhi'
city_menu = ctk.CTkOptionMenu(fields_frame, variable=var_city, values=list(CITY_OPTIONS))
city_menu.grid(row=10, column=1, sticky="ew", padx=10, pady=8)

label_building = ctk.CTkLabel(fields_frame, text="Building Type:")
label_building.grid(row=11, column=0, sticky="w", padx=10, pady=8)
var_building = ctk.StringVar(value=BUILDING_TYPE_OPTIONS[0]) # Default
building_menu = ctk.CTkOptionMenu(fields_frame, variable=var_building, values=list(BUILDING_TYPE_OPTIONS))
building_menu.grid(row=11, column=1, sticky="ew", padx=10, pady=8)

# --- Predict Button ---
predict_button = ctk.CTkButton(main_frame, text="Predict Consumption", command=predict_energy, font=ctk.CTkFont(size=16, weight="bold"), height=45)
predict_button.pack(pady=25, fill="x", padx=10)

# --- Result Frame ---
result_frame = ctk.CTkFrame(main_frame, corner_radius=10) # Added corner radius
result_frame.pack(fill="x", padx=10)

result_total_label = ctk.CTkLabel(result_frame, text="", font=ctk.CTkFont(size=22, weight="bold"))
result_total_label.pack(pady=(15, 5))

result_kwh_label = ctk.CTkLabel(result_frame, text="Click 'Predict' to see the result", font=ctk.CTkFont(size=12), text_color="gray60")
result_kwh_label.pack()

result_breakdown_header = ctk.CTkLabel(result_frame, text="", font=ctk.CTkFont(size=14, weight="bold", underline=True))
result_breakdown_header.pack(pady=(15, 5))

result_breakdown_label = ctk.CTkLabel(result_frame, text="", font=ctk.CTkFont(family="monospace", size=13)) # Monospace for alignment
result_breakdown_label.pack(pady=(0, 15), anchor="w", padx=25)

root.mainloop()
