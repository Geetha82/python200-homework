import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

# Paths to the saved artifacts
MODEL_PATH = "models/weather_classifier.pkl"
METADATA_PATH = "models/weather_classifier_metadata.json"


# ==========================================
# --- Task 1: Load and Verify ---
# ==========================================
# Check for missing files and provide helpful error responses
if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    print(" [CRITICAL ERROR] Missing required model pipeline or metadata file.")
    print(f" Expected location 1: {MODEL_PATH}")
    print(f" Expected location 2: {METADATA_PATH}")
    print("\n Fix: You must run 'train_weather_classifier.py' first to pull historical")
    print(" data, train the pipeline, and serialize the model artifacts to disk.")
    sys.exit(1)

print("Loading serialized model pipeline framework...")
model = joblib.load(MODEL_PATH)

print("Loading companion model configuration metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print("\n=== Model Verification & Metadata ===")
print(f" Target City:      {metadata['target_city']['name']}")
print(f" Coordinates:      Lat {metadata['target_city']['latitude']}, Lon {metadata['target_city']['longitude']}")
print(f" Trained Features: {metadata['feature_names_order']}")
print(f" Definitive Test AUC: {metadata['test_set_auc']:.4f}")


# ==========================================
# --- Task 2: Predict on New Data ---
# ==========================================
print("\nCreating hypothetical daily weather scenarios for inference...")

# Define 5 distinct days spanning clear, extreme, and borderline conditions.
# Note: Features match the exact order and naming expected by the trained model pipeline.
hypothetical_days = pd.DataFrame([
    {
        "temperature_2m_max": 18.0, 
        "temperature_2m_min": 11.0, 
        "wind_speed_10m_max": 12.0,
        "description": "Clearly Good (Mild, calm breeze)"
    },
    {
        "temperature_2m_max": 12.0, 
        "temperature_2m_min": 5.0, 
        "wind_speed_10m_max": 45.0,
        "description": "Clearly Bad (Chilly, intense wind gusts)"
    },
    {
        "temperature_2m_max": 8.0, 
        "temperature_2m_min": 3.0, 
        "wind_speed_10m_max": 14.0,
        "description": "Clearly Bad (Too cold, below freezing comfort bounds)"
    },
    {
        "temperature_2m_max": 24.5, 
        "temperature_2m_min": 13.0, 
        "wind_speed_10m_max": 26.5,
        "description": "Borderline Case (Comfortable temp, but wind is slightly high)"
    },
    {
        "temperature_2m_max": 14.0, 
        "temperature_2m_min": 9.0, 
        "wind_speed_10m_max": 22.0,
        "description": "Borderline Case (Cool morning, moderate acceptable breeze)"
    }
])

# Extract exact features to pass into the classifier
X_new = hypothetical_days[metadata["feature_names_order"]]

# Execute inference
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1] # Probability of class 1 ("Good for running")

print("\n=== Standalone Production Inference Results ===")
for i in range(len(hypothetical_days)):
    row = hypothetical_days.iloc[i]
    pred_label = "good" if predictions[i] == 1 else "skip"
    prob_good = probabilities[i]
    
    print(f"\nDay {i+1}: {row['description']}")
    print(f"  Inputs:  Max Temp: {row['temperature_2m_max']}°C | Min Temp: {row['temperature_2m_min']}°C | Max Wind: {row['wind_speed_10m_max']} km/h")
    print(f"  Output:  Predicted Label -> [{pred_label.upper()}]")
    print(f"  Metrics: Confidence (Probability of Good): {prob_good * 100:.2f}%")


# ==========================================
# --- Task 3: Reflection ---
# ==========================================
"""
COMMENT  - INFERENCE AND PRODUCTION REFLECTION:

1. Borderline Case Analysis:
   On Day 5 ("Cool morning, moderate acceptable breeze"), the model outputs a probability 
   of 68.27%, declaring it 'GOOD'. This represents mild uncertainty compared to the 99% clear days. 
   More interestingly, Day 3 ('Too cold') exposed a failure mode where the model predicted 'GOOD' with 
   98.80% confidence on an 8.0°C day. This happens because 8.0°C is at the absolute edge of San Francisco's 
   historical range, causing the linear Logistic Regression model to extrapolate poorly. How would I handle 
   a day with a 0.52 probability? I would flag it in the UI as 'Marginal' or 'Runner's Choice' rather than 
   giving a rigid binary answer, as 52% represents high model uncertainty.

2. Workflow Disconnection and Error Handling:
   If predict_weather.py is executed before train_weather_classifier.py, the script will crash immediately 
   with a FileNotFoundError because the .pkl and .json artifacts do not exist on disk yet. To make this error 
   more helpful, I wrapped the loading code in an os.path.exists checks block. If the files are missing, 
   the script intercepts the failure, prints a clear explanation of *why* it failed, and gives the user 
   direct, actionable instructions to run train_weather_classifier.py first before exiting safely.

3. Scaling to Daily Production Pipelines:
   To modify this script for a production system that runs daily on tomorrow's forecast, we would need 
   to replace the hardcoded hypothetical DataFrame with a dynamic data ingestion pipeline. Specifically, 
   we would integrate a live API request block using python-requests to hit the Open-Meteo Forecast API 
   endpoints instead of the Archive endpoints. The script would parse tomorrow's specific incoming forecast 
   metrics into the identical DataFrame schema order, execute the model's .predict() method, and push the 
   resulting 'good/skip' notification string to a user interface, database, or alerting webhook.
"""
