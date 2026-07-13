# =====================================================================
# Pre-preprocessing Observations
# =====================================================================
# Observation: When viewing the raw student_performance_math.csv file in a plain 
# text editor, fields are separated by semicolons (;) rather than commas (,). 
# String values are wrapped in quotation marks while numeric attributes are raw. 
# G1, G2, and G3 appear as raw integer columns. Beyond the filename, we must specify 
# the parameter `sep=';'` in pd.read_csv() to avoid parsing errors.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    print("--- Project 02: Student Math Performance ---")
    
    # 1. Clear out path targets using local directory context
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "student_performance_math.csv")
    
    # This points directly to the outputs/ directory under assignments_02/
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # Task 1: Load and Explore
    # =====================================================================
    print("\n--- Task 1: Loading and Exploring Dataset ---")
    df = pd.read_csv(csv_path, sep=';')
    
    print("Dataset Shape (Rows, Columns):", df.shape)
    print("\nFirst Five Rows:")
    print(df.head())
    print("\nData Types of All Columns:")
    print(df.dtypes)
    
    plt.figure()
    plt.hist(df['G3'], bins=21, range=(0, 20), color='skyblue', edgecolor='black')
    plt.title("Distribution of Final Math Grades")
    plt.xlabel("Final Math Grade (G3)")
    plt.ylabel("Number of Students")
    plt.savefig(os.path.join(output_dir, "g3_distribution.png"))
    plt.close()

    # =====================================================================
    # Task 2: Preprocess the Data
    # =====================================================================
    print("\n--- Task 2: Preprocessing the Data ---")
    df_clean = df[df['G3'] > 0].copy()
    print("Shape before filtering out G3=0:", df.shape)
    print("Shape after filtering out G3=0:", df_clean.shape)

    binary_cols = ['schoolsup', 'internet', 'higher', 'activities']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map({'yes': 1, 'no': 0})
    df_clean['sex'] = df_clean['sex'].map({'F': 0, 'M': 1})
    
    corr_original = df['absences'].corr(df['G3'])
    corr_filtered = df_clean['absences'].corr(df_clean['G3'])
    print("\nPearson correlation between absences and G3:")
    print("Original Dataset correlation:", corr_original)
    print("Filtered Dataset correlation:", corr_filtered)

    # =====================================================================
    # Task 3: Exploratory Data Analysis
    # =====================================================================
    print("\n--- Task 3: Exploratory Data Analysis ---")
    all_numeric_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", 
                        "internet", "sex", "freetime", "activities", "traveltime", "absences", "G3"]
    correlations = df_clean[all_numeric_cols].corr()["G3"].sort_values()
    print("\nPearson correlation coefficients with G3 (Sorted):")
    print(correlations)

    plt.figure()
    plt.scatter(df_clean["failures"], df_clean["G3"], alpha=0.5, color="crimson")
    plt.title("Final Grade vs Past Class Failures")
    plt.xlabel("Number of Past Class Failures")
    plt.ylabel("Final Math Grade (G3)")
    plt.savefig(os.path.join(output_dir, "failures_vs_g3.png"))
    plt.close()

    plt.figure()
    df_clean.boxplot(column="G3", by="studytime")
    plt.title("Final Grade Distribution by Weekly Study Time")
    plt.suptitle("")
    plt.xlabel("Study Time Category (1=Low to 4=High)")
    plt.ylabel("Final Math Grade (G3)")
    plt.savefig(os.path.join(output_dir, "studytime_boxplot.png"))
    plt.close()

    # =====================================================================
    # Task 4: Baseline Model
    # =====================================================================
    print("\n--- Task 4: Baseline Model (Failures Only) ---")
    X_base = df_clean[["failures"]].values
    y_base = df_clean["G3"].values
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
    
    model_base = LinearRegression()
    model_base.fit(X_train_b, y_train_b)
    y_pred_b = model_base.predict(X_test_b)
    
    print("Baseline Slope:", model_base.coef_)
    print("Baseline Test RMSE:", np.sqrt(np.mean((y_pred_b - y_test_b) ** 2)))
    print("Baseline Test R²:", model_base.score(X_test_b, y_test_b))

    # =====================================================================
    # Task 5: Build the Full Model
    # =====================================================================
    print("\n--- Task 5: Building the Full Model ---")
    feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", 
                    "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]
    X_full = df_clean[feature_cols].values
    y_full = df_clean["G3"].values
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    model_full = LinearRegression()
    model_full.fit(X_train_f, y_train_f)
    y_pred_f = model_full.predict(X_test_f)
    
    print("Train R²:", model_full.score(X_train_f, y_train_f))
    print("Test R² :", model_full.score(X_test_f, y_test_f))
    print("Full Model Test RMSE:", np.sqrt(np.mean((y_pred_f - y_test_f) ** 2)))
    print("\nFeature Coefficients:")
    for name, coef in zip(feature_cols, model_full.coef_):
        print(f"{name:12s}: {coef:+.3f}")

    # =====================================================================
    # Task 6: Evaluate and Summarize
    # =====================================================================
    print("\n--- Task 6: Evaluate and Summarize ---")
    plt.figure()
    plt.scatter(y_pred_f, y_test_f, color="purple", alpha=0.7)
    all_vals = np.concatenate([y_pred_f, y_test_f])
    min_val, max_val = all_vals.min(), all_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], color="darkorange", linestyle="--")
    plt.title("Predicted vs Actual (Full Model)")
    plt.xlabel("Predicted Grade (y_hat)")
    plt.ylabel("True Grade (y)")
    plt.savefig(os.path.join(output_dir, "predicted_vs_actual.png"))
    plt.close()

    # =====================================================================
    # Neglected Feature: Testing the Power of G1
    # =====================================================================
    print("\n--- Neglected Feature: Testing the Power of G1 ---")
    power_features = feature_cols + ["G1"]
    X_pow = df_clean[power_features].values
    y_pow = df_clean["G3"].values
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pow, y_pow, test_size=0.2, random_state=42)
    
    model_power = LinearRegression()
    model_power.fit(X_train_p, y_train_p)
    print("New Test R² with G1 Included:", model_power.score(X_test_p, y_test_p))

if __name__ == "__main__":
    main()
