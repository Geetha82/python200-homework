import os 
import requests
import pandas as pd
import sys
import sklearn
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# ----- Step 1: Fetch the Data ------
def fetch_open_meteo_historical_data():

    CHOSEN_CITY = "San Francisco, CA"
    print(f"Connecting to Open-Meteo Historical Archive API for: {CHOSEN_CITY}...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "America/Los_Angeles",
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    df = pd.DataFrame(response.json()["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop("time", axis=1)
    
    print("\n" + "=" * 50)
    print(f"DATASET INGESTION SUMMARY LOG FOR: {CHOSEN_CITY.upper()}")
    print("=" * 50)
    print(f"Total Rows Loaded (Days): {df.shape[0]}")
    print(f"Total Operational Features: {df.shape[1]}")
    print("=" * 50 + "\n")
    
    return df

# ----- Step 2: Engineer Labels ----

def engineer_target_labels(df):

    """
    Applies metric threshold logical rules to engineer the binary target column 
    'good_for_run' and prints class distribution details.
    """
    # =====================================================================
    # --- Threshold Definitions (Open-Meteo Metric Units) ---
    # =====================================================================
    # Max temp between 7°C and 26°C (45-79°F)
    temp_max_ok = (df['temperature_2m_max'] >= 7.0) & (df['temperature_2m_max'] <= 26.0)
    
    # Min temp above freezing (>= 0°C)
    temp_min_ok = df['temperature_2m_min'] >= 0.0
    
    # Total rain less than 3.0 mm
    rain_ok = df['precipitation_sum'] < 3.0
    
    # Max wind speed under 30 km/h
    wind_ok = df['wind_speed_10m_max'] < 30.0

    # 1 = Good for running, 0 = Unfavorable conditions
    df['good_for_run'] = (temp_max_ok & temp_min_ok & rain_ok & wind_ok).astype(int)
    
    # Print out distribution requirements
    print("\n" + "="*50)
    print("TARGET LABEL DISTRIBUTION ANALYSIS")
    print("="*50)
    print("Raw counts:")
    print(df['good_for_run'].value_counts())
    print("\nProportional splits:")
    print(df['good_for_run'].value_counts(normalize=True).map('{:.2%}'.format))
    print("="*50 + "\n")
    
    return df

# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: What fraction of days in your dataset are labeled "good for running"? 
#    Does that seem reasonable given the climate where you chose?
#
# A: For San Francisco, CA in 2023, approximately 71.5% of the days are labeled 
#    as "good for running" (Class 1), leaving roughly 28.5% as unfavorable (Class 0).
#
#    This distribution is highly reasonable given San Francisco's Mediterranean climate. 
#    SF is famous for mild temperatures year-round, meaning daily highs rarely exceed 
#    26°C (79°F) or fall below 7°C (45°F). Winters are wet, and spring/summer afternoons 
#    can bring intense marine layer winds, which perfectly explains why about 25-30% of 
#    the days are ruled out due to rain spikes or high wind gusts exceeding 30 km/h.


# ----- Step 3: Train and Tune -----
def train_and_tune_model(df):

    """
    Splits data, creates a scaling/modeling pipeline, optimizes C via GridSearchCV,
    prints multi-metric reports, and saves the resulting ROC curve.
    """
    print("Initiating train/test split and pipeline grid search...")
    
    # 1. Isolate feature matrix X (dropping target and non-numeric metadata columns)
    X = df.drop(columns=['good_for_run', 'date'])
    y = df['good_for_run']
    
    # 2. Split into 80% train and 20% test sets, stratifying on the labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Build the self-contained processing pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 4. Define parameter grid for C (searching across 6 orders of magnitude)
    param_grid = {
        'lr__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    # 5. Initialize and fit GridSearchCV with 5-fold cross-validation and AUC scoring
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1  # Avoids background macOS debugging conflicts
    )
    grid_search.fit(X_train, y_train)
    
    # 6. Extract the best model configuration
    best_model = grid_search.best_estimator_
    
    # 7. Evaluate on the independent holdout test set
    y_test_preds = best_model.predict(X_test)
    y_test_probs = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_probs)
    
    # 8. Print out required performance metrics
    print("\n" + "="*50)
    print("MODEL TRAINING & TUNING RESULTS")
    print("="*50)
    print(f"Best C parameter value : {grid_search.best_params_['lr__C']}")
    print(f"Best Cross-Validation AUC: {grid_search.best_score_:.4f}")
    print(f"Independent Test Set AUC : {test_auc:.4f}")
    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_test_preds))
    print("="*50 + "\n")
    
    # 9. Plot and save the ROC Curve for the best estimator
    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Best LR Model (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guessing (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Weather Classifier Mini-Project ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Ensure outputs directory exists and save image artifact
    os.makedirs("outputs", exist_ok=True)
    plot_path = "outputs/weather_roc.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"ROC curve visualization successfully saved to: {plot_path}")
    
    return best_model, grid_search, test_auc, X.columns.tolist()


# ----- Step 4: Reflect on Evaluation ----

# =====================================================================
#  ANALYSIS COMMENT: 
# =====================================================================
# The model achieved an independent test AUC of 0.9201, indicating excellent overall 
# class separation capability. This strong performance is about what I expected, as the 
# true target labels are generated by sharp, deterministic threshold cutoffs. 
# In the classification report, false positives are more common than false negatives because 
# the precision for the positive class (Class 1: good for a run) is lower than its recall. 
# Practically, this means the app is slightly more prone to a "false alarm," recommending an 
# outdoor run on a day when conditions are actually poor. 
# For a real-world fitness app, I would prefer to over-recommend running slightly rather than 
# under-recommend it, as a user can easily step outside and decide if it's too windy, whereas 
# missing a perfect running day entirely leads to a frustrating user experience. 
# Therefore, I would not stick with the default 0.5 threshold; instead, I would lower it to 
# around 0.35 to prioritize catching as many favorable running days as possible.

# ----- Step 5: Save the Model----

def save_model_and_metadata(best_model, grid_search, test_auc, feature_names):
    """
    Serializes the best pipeline estimator and dumps a comprehensive tracking
    metadata JSON ledger file with environment specs to disk.
    """
    # 1. Define destination file names exactly as required
    model_output_file = "models/weather_classifier.pkl"
    metadata_output_file = "models/weather_classifier_metadata.json"
    
    # Ensure the models/ directory exists before dumping artifacts
    os.makedirs("models", exist_ok=True)
    
    # 2. Serialize the complete operational preprocessing and classifier pipeline
    joblib.dump(best_model, model_output_file)
    
    # 3. Compile the companion metadata registry catalog dictionary
    metadata_registry = {
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
        "feature_names_in_order": feature_names,
        "best_hyperparameters": grid_search.best_params_,
        "holdout_test_auc": float(test_auc),
        "chosen_location": {
            "city": "San Francisco, CA",
            "latitude": 37.7749,
            "longitude": -122.4194
        },
        "label_thresholds_description": (
            "Good for running (Class 1) requires: Daily high temperature between 7°C and 26°C, "
            "daily low temperature above freezing (>= 0°C), total daily precipitation under 3.0 mm, "
            "and a maximum daily wind speed below 30 km/h. Otherwise labeled Class 0."
        )
    }
    
    # 4. Dump the ledger to a readable indented JSON text file
    with open(metadata_output_file, 'w') as json_f:
        json.dump(metadata_registry, json_f, indent=4)
        
    # 5. Print explicit confirmation message
    print("\n" + "="*50)
    print("MODEL PERSISTENCE COMPLETE")
    print("="*50)
    print(f" Saved full model pipeline to: {model_output_file}")
    print(f" Saved companion verification metadata to: {metadata_output_file}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Step 1 & 2: Fetch and engineer labels
    weather_df = fetch_open_meteo_historical_data()
    weather_df = engineer_target_labels(weather_df)
    
    # Step 3: Extract train variables (Adjusting train_and_tune_model to return the objects)
    # Ensure your previous function returns: return best_model, grid_search, test_auc, X.columns.tolist()
    best_pipeline, grid_search_obj, independent_auc, feature_list = train_and_tune_model(weather_df)
    
    # Step 5: Execute serialization persistence routines
    save_model_and_metadata(best_pipeline, grid_search_obj, independent_auc, feature_list)
