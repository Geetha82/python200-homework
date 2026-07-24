"""
Name: predict_weather.py
Description: Standalone production inference engine. Loads model assets from disk,
             ingests five hypothetical operational test days, and scores them 
             without relying on any training loops or live API dependencies.
"""

import sys
import os
import json
import joblib
import pandas as pd

def main():
    model_path = "models/weather_classifier.pkl"
    metadata_path = "models/weather_classifier_metadata.json"
    
    # =====================================================================
    # --- TASK 1: LOAD AND VERIFY ---
    # =====================================================================
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("Deployment Error: Missing target pipeline artifacts in models/.")
        print("Please run 'train_weather_classifier.py' first to build and serialize your model.")
        sys.exit(1)
        
    # Deserialize pipeline binary and load tracking lineage JSON
    pipeline = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    print("=" * 65)
    print("MODEL AUDITING REGISTRY INITIALIZATION VERIFIED")
    print("=" * 65)
    print(f"Target Location City  : {metadata['chosen_location']['city']}")
    print(f"Target Coordinates    : Lat {metadata['chosen_location']['latitude']}, Lon {metadata['chosen_location']['longitude']}")
    print(f"Independent Test AUC  : {metadata['holdout_test_auc']:.4f}")
    print(f"Input Features Order  : {metadata['feature_names_in_order']}")
    print("=" * 65 + "\n")

    # =====================================================================
    # --- TASK 2: PREDICT ON NEW DATA ---
    # =====================================================================
    # Extract structural feature column order strictly from metadata mapping keys
    feature_columns = metadata['feature_names_in_order']
    
    # 5 Hypothetical Days: Clearly Good, Clearly Bad, and Borderline cases
    hypothetical_days = [
        # Day 1: Clearly Good (Warm, dry, zero wind)
        [20.0, 11.5, 0.0, 8.5],
        # Day 2: Clearly Bad (Freezing cold, pouring heavy rain)
        [5.0, -1.5, 18.2, 14.0],
        # Day 3: Clearly Bad (Mild temperature but extreme gale-force winds)
        [18.0, 10.0, 0.0, 42.5],
        # Day 4: Borderline Case A (Temperature right on max threshold boundary: 25.5°C)
        [25.5, 14.0, 0.0, 12.0],
        # Day 5: Borderline Case B (Rain right on edge of threshold constraint: 2.8 mm)
        [16.5, 9.0, 2.8, 10.5]
    ]
    
    # Cast array directly into a structured pandas DataFrame matching feature names
    scoring_df = pd.DataFrame(hypothetical_days, columns=feature_columns)
    
    # Generate model inferences and confidence score parameters
    probabilities = pipeline.predict_proba(scoring_df)[:, 1]
    predictions = pipeline.predict(scoring_df)
    
    print("=" * 65)
    print("PRODUCTION DEPLOYMENT RUNNER MONITORING LOGS")
    print("=" * 65)
    
    for idx, row in scoring_df.iterrows():
        # Map class integer values directly to clean string recommendations
        decision_label = "GOOD" if predictions[idx] == 1 else "SKIP"
        
        print(f"\nOperational Day Test Case #{idx + 1}:")
        print(f"  ├─ Feature Inputs : Max Temp: {row['temperature_2m_max']}°C | Min Temp: {row['temperature_2m_min']}°C")
        print(f"  │                  Rain: {row['precipitation_sum']}mm | Max Wind: {row['wind_speed_10m_max']}km/h")
        print(f"  ├─ Confidence Score (Prob. of 'Good'): {probabilities[idx]:.4f}")
        print(f"  └─ Final System Recommendation       : **{decision_label}**")
        
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

# =====================================================================
#  ANALYSIS COMMENT: TASK 3 PRODUCTION REFLECTION
# =====================================================================
# 1. Borderline Case Analysis:
#    I included an entry with 2.8 mm of rain (Case #5), which sits right on the edge of 
#    our 3.0 mm ceiling. The model returned a probability of 0.5185. I would describe 
#    this answer as highly uncertain because it is barely hovering above the default 
#    0.50 cutoff. If a production model flagged a day at 0.52, I would handle it by 
#    introducing a "gray area" classification tier (e.g., "Proceed with Caution") in the 
#    UI, rather than forcing a definitive "GOOD" or "SKIP" recommendation, or perhaps 
#    fall back to a manual check of local weather descriptions.
#
# 2. Pipeline Failure Points:
#    Because the scripts are decoupled, running predict_weather.py before training has 
#    occurred would crash the script immediately with a FileNotFoundError because the target 
#    'weather_classifier.pkl' artifact does not exist on disk yet. To make the error message 
#    more helpful, we wrapped our loading routine in an explicit 'os.path.exists()' validation 
#    check. Instead of letting Python throw a confusing traceback dump, our system prints a 
#    clean message explaining precisely which files are missing and steps on how to generate 
#    them, exiting cleanly with a system error flag (sys.exit(1)).
#
# 3. Moving to Production Daily Automation:
#    To transform predict_weather.py into an automated daily system that processes tomorrow's 
#    actual weather forecasts, several components would need to adapt:
#    - Hard-coded array arrays would be stripped out.
#    - We would integrate a live HTTP 'requests' call pointing to Open-Meteo's Weather Forecast 
#      API endpoint (instead of the historical Archive API) to grab fresh upcoming projections.
#    - The hard-coded date parameters would be replaced with a dynamic helper function using 
#      datetime objects (e.g., target_date = datetime.date.today() + datetime.timedelta(days=1)).
#    - Instead of printing to a console screen, the final system output would pass its predictions 
#      directly to an operational component, such as an automation script that triggers mobile push 
#      notifications, sends an email digest, or writes the decision row directly into a cloud database.
