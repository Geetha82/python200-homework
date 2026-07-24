import json
import os
import sys
import joblib
import pandas as pd

MODEL_PATH = "models/weather_classifier.pkl"
METADATA_PATH = "models/weather_classifier_metadata.json"

# Task 1: Load and Verify
if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    print(f"[CRITICAL ERROR] Missing required model pipeline or metadata file.")
    print(
        f"Fix: Run 'train_weather_classifier.py' first to serialize artifacts to disk."
    )
    sys.exit(1)

print("Loading model pipeline and companion metadata...")
model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# Dynamically extract feature order from metadata
expected_features = metadata["feature_names_order"]

print("\n=== Model Verification & Metadata ===")
print(f"Target City:       {metadata['target_city']['name']}")
print(f"Expected Features: {', '.join(expected_features)}")
print(f"Model Test AUC:    {metadata['test_set_auc']:.4f}\n")


# Task 2: Predict on New Data (Aligned with 4-variable spec)
hypothetical_days = pd.DataFrame(
    [
        {
            "temperature_2m_max": 18.0,
            "temperature_2m_min": 11.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 12.0,
            "description": "Clearly Good (Mild, calm breeze)",
        },
        {
            "temperature_2m_max": 12.0,
            "temperature_2m_min": 5.0,
            "precipitation_sum": 12.5,
            "wind_speed_10m_max": 45.0,
            "description": "Clearly Bad (Chilly, pouring rain, intense wind gusts)",
        },
        {
            "temperature_2m_max": 3.0,
            "temperature_2m_min": -4.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 14.0,
            "description": "Clearly Bad (Too cold, below freezing comfort bounds)",
        },
        {
            "temperature_2m_max": 24.5,
            "temperature_2m_min": 13.0,
            "precipitation_sum": 2.8,
            "wind_speed_10m_max": 26.5,
            "description": "Borderline Case (Comfortable temp, but rain and wind are near thresholds)",
        },
        {
            "temperature_2m_max": 14.0,
            "temperature_2m_min": 9.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 22.0,
            "description": "Borderline Case (Cool morning, moderate acceptable breeze)",
        },
    ]
)

# Enforce feature validation matching the metadata schema
X_new = hypothetical_days[expected_features]

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

print("=== Standalone Production Inference Results ===")
for i in range(len(hypothetical_days)):
    row = hypothetical_days.iloc[i]
    pred_label = "good" if predictions[i] == 1 else "skip"
    prob_good = probabilities[i]

    print(f"\nDay {i+1}: {row['description']}")
    print(
        f"  Inputs:     Max Temp: {row['temperature_2m_max']}°C | "
        f"Min Temp: {row['temperature_2m_min']}°C | "
        f"Precipitation: {row['precipitation_sum']}mm | "
        f"Max Wind: {row['wind_speed_10m_max']}km/h"
    )
    print(f"  Output:     Predicted Label -> [{pred_label.upper()}]")
    print(f"  Metrics:    Confidence (Probability of Good): {prob_good * 100:.2f}%")
    print("-" * 70)

# --- Task 3: Reflection ---
"""
REFLECTION COMMENT:

1. Borderline and Extreme Case Analysis:
Day 3 highlights a severe mathematical limitation of linear classifiers. Despite being a freezing day (3.0°C Max / -4.0°C Min), the model outputs a 99.98% probability of being GOOD. This happens because Logistic Regression relies on linear combinations of features and cannot naturally create bounded 'if-else' box constraints (like our strict >= 4.0°C minimum temperature rule). It extrapolates the lower temperature values to mean an increasingly ideal running environment, entirely missing the safety floor cutoff. For a true borderline day showing a 0.52 probability, the production app should display a 'Marginal Conditions' status flag to the user rather than forcing a blunt binary decision.

2. Workflow Disconnection:
Running 'predict_weather.py' before 'train_weather_classifier.py' breaks the execution chain because the underlying serialized .pkl pipeline and .json metadata companion artifacts do not exist on disk yet. To improve user experience, this script uses an upfront os.path.exists check to catch missing dependencies early, hide raw unhandled Python tracebacks, and print clean instructions to run training first.

3. Production Scaling:
To scale this script into a production service capable of serving daily forecast predictions, we must replace the static, hard-coded hypothetical DataFrame with a dynamic data-fetching step. Every morning, the script would make a live HTTP request targeting the 7-day or tomorrow forecast endpoint of the Open-Meteo API. The incoming JSON response must then be converted into a pandas DataFrame that strictly enforces the same feature schema order found in the metadata file: 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', and 'wind_speed_10m_max'. Preserving this identical column order and structural data schema ensures that the live forecast data aligns perfectly with the model coefficients without throwing matrix dimension errors or corrupting calculations during inference. Finally, we would schedule this script to run automatically using an orchestrator like Airflow or a cron job, sending the final labels directly to an external database or user notification webhook.
"""
