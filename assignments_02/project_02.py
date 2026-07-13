# =====================================================================
# Pre-preprocessing Observations
# =====================================================================
# When viewing the raw student_performance_math.csv file in a plain 
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
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
# ===== TTask 1: Task 1: Load and Explore =====

print("\n===== Task 1: Task 1: Load and Explore =====")

# --- 1. Load and Explore ---
# Load the dataset with the correct custom separator
df = pd.read_csv("student_performance_math.csv", sep=";")

# Print the foundational shape of the raw dataset
print("--- Dataset Shape ---")
print(df.shape)
print()

# Print the first five rows for manual column inspection
print("--- First Five Rows ---")
print(df.head())
print()

# Print the explicit data types of all 18 trimmed columns
print("--- Column Data Types ---")
print(df.dtypes)
print()

# --- 2. Data Visualization ---
# Ensure the outputs directory exists before saving
os.makedirs("outputs", exist_ok=True)

# Plot a histogram of G3 with 21 bins (0 to 20 inclusive)
plt.figure(figsize=(8, 5))
plt.hist(df['G3'], bins=21, range=(0, 20), edgecolor='black', color='royalblue', alpha=0.8)

# Format the plot according to the structural requirements
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Math Grade (G3)")
plt.ylabel("Number of Students")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust visual boundaries and save directly to the designated outputs file
plt.tight_layout()
plt.savefig("outputs/g3_distribution.png")
plt.close()

print("Plot successfully saved to outputs/g3_distribution.png")


# ===== Task 2: Preprocess the Data =====
print("--- Task 2: Data Cleaning & Correlation Analysis ---")

# 1. Filter out G3=0 rows
# Reasoning: Students with G3=0 missed the final exam rather than earning a true 
# score of zero. Keeping these rows distorts a linear regression model because 
# it forces the model to treat a behavioral administrative event (absenteeism) 
# as a numerical academic outcome, creating an artificial bimodal distribution 
# that ruins linear assumptions.
df_filtered = df[df['G3'] > 0].copy()

print(f"Shape of dataset BEFORE filtering out zeros: {df.shape}")
print(f"Shape of dataset AFTER filtering out zeros:  {df_filtered.shape}")
print(f"Total rows removed (absentee students): {df.shape[0] - df_filtered.shape[0]}\n")

# 2. Convert binary columns to numeric features
yes_no_cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in yes_no_cols:
    df_filtered[col] = df_filtered[col].map({'yes': 1, 'no': 0})

df_filtered['sex'] = df_filtered['sex'].map({'F': 0, 'M': 1})

# 3. Compute Pearson correlation between absences and G3
corr_original = df['absences'].corr(df['G3'])
corr_filtered = df_filtered['absences'].corr(df_filtered['G3'])

print(f"Pearson correlation (Absences vs G3) on ORIGINAL data: {corr_original:.4f}")
print(f"Pearson correlation (Absences vs G3) on FILTERED data: {corr_filtered:.4f}\n")

# Explanation of the correlation difference:
# In the original data, students who dropped or missed the exam entirely ended up 
# with a G3 of 0. Crucially, many of these disengaged or dropping students had 
# extremely high absence rates. 
# By matching very high absences with a hard grade of 0, the raw data artificially 
# implies a steep, massive downward performance crash. Once these exam dropouts are 
# removed, we see the relationship among exam-taking students: a much weaker, more 
# normal real-world correlation where incremental absences only slightly nudge 
# performance downward.


# ===== Task 3: Exploratory Data Analysis =====

print("--- Task 3: Exploratory Data Analysis ---")

numeric_features = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
    'failures', 'absences', 'freetime', 'goout', 'Walc'
]

# Calculate and sort Pearson correlation with G3 on the filtered dataset
correlations = df_filtered[numeric_features].corrwith(df_filtered['G3'])
sorted_correlations = correlations.sort_values()

print("Numeric Feature Correlations with G3 (Sorted):")
print(sorted_correlations)
print()

os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------------------------
# Plot A: Box Plot of G3 grouped by Past Class Failures
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
df_filtered.boxplot(column='G3', by='failures', grid=False, 
                    patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title("Final Math Grade (G3) Distribution by Past Failures")
plt.suptitle("") 
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("outputs/g3_by_failures_box.png")
plt.close()

# PLOT A VISUAL ANALYSIS VERIFICATION:
# This plot tracks the strongest negative feature: 'failures' (~-0.29).
# Visually, the median G3 grade drops continuously at each incremental step 
# of past failures, establishing it as a primary negative threshold feature.

# ------------------------------------------------------------------------------
# Plot B: Bar Plot of Mean G3 by Mother's Education Level (Medu)
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
mean_g3_by_medu = df_filtered.groupby('Medu')['G3'].mean()
mean_g3_by_medu.plot(kind='bar', color='seagreen', edgecolor='black', alpha=0.8)
plt.title("Average Final Math Grade by Mother's Education Level")
plt.xlabel("Mother's Education Level (0=None to 4=Higher Ed)")
plt.ylabel("Mean Final Grade (G3)")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/g3_by_medu_bar.png")
plt.close()

# PLOT B VISUAL ANALYSIS VERIFICATION:
# This plot tracks the strongest positive feature: 'Medu' (~+0.24).
# Visually, the mean G3 grade increases almost linearly with maternal education levels, 
# demonstrating that higher socioeconomic anchors correspond to higher student scores.


# ==============================================================================
# --- Task 5: Build the Full Model ---
# ==============================================================================
print("--- Task 5: Full Multiple Regression Model ---")

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", 
    "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"
]

X_full = df_filtered[feature_cols].values
y_full = df_filtered["G3"].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

full_model = LinearRegression()
full_model.fit(X_train_f, y_train_f)

train_r2_f = r2_score(y_train_f, full_model.predict(X_train_f))
test_r2_f = r2_score(y_test_f, full_model.predict(X_test_f))
test_rmse_f = root_mean_squared_error(y_test_f, full_model.predict(X_test_f))

print(f"Full Model Train R² Score: {train_r2_f:.4f}")
print(f"Full Model Test R² Score:  {test_r2_f:.4f}")
print(f"Full Model Test RMSE:      {test_rmse_f:.4f}\n")

# EXACT OUTPUT PATTERN REQUIRED BY ASSIGNMENT INSTRUCTIONS:
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
print()

# Coefficient Insight: 'Fedu' turns negative (-0.147) because of multicollinearity 
# with 'Medu'. 'schoolsup' is negative (-1.352) due to reverse causality—remedial 
# tutoring is target-assigned to students who enter already struggling.
# Train vs Test Gap: The metrics (0.235 vs 0.201) are tight, meaning no overfitting.
# Production Deployment Choice: Keep actionable items ('failures', 'schoolsup', 
# 'higher', 'Medu', 'studytime'). Drop 'sex' due to ethical fairness constraints, 
# and drop low-value variance columns ('activities', 'internet', 'traveltime').


# ===== Task 4: Baseline Model =====

print("\n--- Task 4: Baseline Model (Failures Only) ---")

# 1. Isolate the baseline single feature and target
X_baseline = df_filtered[['failures']]
y_baseline = df_filtered['G3']

# 2. Split into training and test sets (80/20 split)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42
)

# 3. Fit the baseline Linear Regression model
baseline_model = LinearRegression()
baseline_model.fit(X_train_b, y_train_b)

# 4. Generate predictions on the validation test set
test_preds_b = baseline_model.predict(X_test_b)

# 5. Calculate explicit evaluation metrics
slope_b = baseline_model.coef_[0]
rmse_b = root_mean_squared_error(y_test_b, test_preds_b)
r2_b = r2_score(y_test_b, test_preds_b)

print(f"Baseline Slope (Coefficient): {slope_b:.4f}")
print(f"Baseline Test RMSE:            {rmse_b:.4f}")
print(f"Baseline Test R² Score:         {r2_b:.4f}\n")

# Interpretation Comments:
#
# Slope Interpretation:
# The slope is roughly -1.13. In plain English, this means that for every additional 
# past class failure a student has on their record, their predicted final math grade 
# drops by approximately 1.13 points on the 0-20 scale, assuming everything else is equal.
#
# RMSE Interpretation:
# The Root Mean Squared Error sits around 2.92. On a 0-20 scale, this means the 
# model's grade predictions miss the students' actual final math grades by an 
# average margin of roughly 3 points. 
#
# R² Score Interpretation:
# The R² score is roughly 0.08 (explaining about 8% of the variance in grades). 
# This is slightly worse than or right in line with expectations from our exploratory 
# analysis. While 'failures' had the highest relative correlation (~ -0.29), squaring 
# that correlation coefficient gives an expected baseline variance explanation of around 
# 8-9%. This low score proves that a student's past failure history alone leaves 92% 
# of their final academic performance unexplained, highlighting the desperate need for 
# a multiple regression model.


# ===== Task 5: Build the Full Model =====
print("--- Task 5: Full Multiple Regression Model ---")

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", 
    "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"
]

X_full = df_filtered[feature_cols].values
y_full = df_filtered["G3"].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

full_model = LinearRegression()
full_model.fit(X_train_f, y_train_f)

train_r2_f = r2_score(y_train_f, full_model.predict(X_train_f))
test_r2_f = r2_score(y_test_f, full_model.predict(X_test_f))
test_rmse_f = root_mean_squared_error(y_test_f, full_model.predict(X_test_f))

print(f"Full Model Train R² Score: {train_r2_f:.4f}")
print(f"Full Model Test R² Score:  {test_r2_f:.4f}")
print(f"Full Model Test RMSE:      {test_rmse_f:.4f}\n")

# EXACT OUTPUT PATTERN REQUIRED BY ASSIGNMENT INSTRUCTIONS:
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
print()

# Coefficient Insight: 'Fedu' turns negative (-0.147) because of multicollinearity 
# with 'Medu'. 'schoolsup' is negative (-1.352) due to reverse causality—remedial 
# tutoring is target-assigned to students who enter already struggling.
# Train vs Test Gap: The metrics (0.235 vs 0.201) are tight, meaning no overfitting.
# Production Deployment Choice: Keep actionable items ('failures', 'schoolsup', 
# 'higher', 'Medu', 'studytime'). Drop 'sex' due to ethical fairness constraints, 
# and drop low-value variance columns ('activities', 'internet', 'traveltime').



# ===== Task 6: Evaluate and Summarize =====
print("--- Task 6: Evaluate and Summarize ---")

plt.figure(figsize=(7, 7))
plt.scatter(full_model.predict(X_test_f), y_test_f, color='darkorange', edgecolor='k', alpha=0.7)
min_val = min(min(full_model.predict(X_test_f)), min(y_test_f))
max_val = max(max(full_model.predict(X_test_f)), max(y_test_f))
plt.plot([min_val, max_val], [min_val, max_val], color='navy', linestyle='--')
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Grade (ŷ)")
plt.ylabel("True Final Grade (y)")
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# Plot Diagnostic Insight:
# A value above the diagonal represents an under-prediction; below means over-prediction. 
# The model visibly struggles at both extremes (truncating low and high predictions).

# ------------------------------------------------------------------------------
# CONCISE PLAIN-LANGUAGE SUMMARY
# ------------------------------------------------------------------------------
# * Filtered Dataset Size: 357 student records.
# * Test Dataset Size: 72 student records.
# * Metric Meaning: An R² of 0.20 means lifestyle features explain 20% of grade variance. 
#   An RMSE of 2.72 means typical grade predictions miss reality by ±2.7 points on the 0-20 scale.
# * Largest Positive Feature: 'higher' (+1.10) — wanting university tracks to higher performance.
# * Largest Negative Feature: 'schoolsup' (-1.35) — tutoring flags pre-existing student struggle.
# * Core Surprise: Partying columns ('Walc', 'goout') have a negligible correlation with outcomes.
# ------------------------------------------------------------------------------

# ==============================================================================
# --- Neglected Feature: The Power of G1 ---
# ==============================================================================

print("\n--- Neglected Feature: The Power of G1 ---")

# 1. Update feature selection to include G1 alongside behavioral markers
g1_feature_cols = feature_cols + ["G1"]

# Extract the new updated feature matrix
X_g1 = df_filtered[g1_feature_cols].values
y_g1 = df_filtered["G3"].values

# 2. Split the data using the exact same split structure
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.2, random_state=42
)

# 3. Fit the new Linear Regression model
g1_model = LinearRegression()
g1_model.fit(X_train_g1, y_train_g1)

# 4. Generate predictions and evaluate metrics
test_preds_g1 = g1_model.predict(X_test_g1)
test_r2_g1 = r2_score(y_test_g1, test_preds_g1)
test_rmse_g1 = root_mean_squared_error(y_test_g1, test_preds_g1)

print(f"Model with G1 Test R² Score: {test_r2_g1:.4f}")
print(f"Model with G1 Test RMSE:      {test_rmse_g1:.4f}\n")

# ==============================================================================
# --- CRITICAL QUESTIONS AND INTERPRETATION COMMENTS ---
# ==============================================================================
#
# Does a high R² mean G1 is causing G3?
# No, G1 does not *cause* G3. This is a classic case of proxy/confounding variables 
# rather than direct causation. Both G1 and G3 are separate downstream measurements 
# of the same underlying variable: a student's actual core math proficiency, 
# work ethic, and mastery of the specific curriculum. G1 acts as an early test 
# drive; it correlates highly because it measures identical knowledge under 
# identical conditions, not because it drives the final outcome.
#
# Is this a useful model for identifying students who might struggle?
# Financially and logistically, no. While it is highly accurate at predicting the 
# exact final score, it is not highly actionable for proactive interventions. 
# By the time G1 grades are locked in and finalized, the first trimester of the 
# school year has already concluded. Staggering failures have already accumulated on 
# a student's record, making it a "reactive warning" rather than an "early prediction."
#
# What might educators need to do if they wanted to intervene early?
# If educators want to intervene *before* G1 data exists, they must rely on the 
# lifestyle model built in Task 5. They would need to build a screening pipeline 
# during orientation week tracking demographic and history flags:
# 1. Flag students with a history of past class failures ('failures' column).
# 2. Audit onboarding intake surveys to identify students missing an active 
#    desire to pursue university studies ('higher' column == 0).
# 3. Provide proactive resource routing to students coming from weaker school 
#    districts or lower parental educational backdrops ('Medu').
# By utilizing the lower R² behavioral model on Day 1, schools can allocate 
# tutors weeks before the first midterm exam even takes place.
