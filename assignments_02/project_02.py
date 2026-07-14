# ==============================================================================
# Preprocessing Observation:
# The raw 'student_performance_math.csv' file uses semicolons (';') as a delimiter
# rather than commas, with string values wrapped in double quotes. To correctly
# load this into a DataFrame, the 'sep=";"' parameter must be specified in 
# pd.read_csv().
# ==============================================================================

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from prefect import flow, task, get_run_logger

# ==============================================================================
# TASK 1: LOAD AND EXPLORE
# ==============================================================================

# 1. Load the dataset with the correct semicolon separator
df = pd.read_csv("student_performance_math.csv", sep=";")

# 2. Print initial exploration data to the console
print("--- Task 1: Initial Exploration ---")
print(f"Dataset Shape: {df.shape}\n")

print("First 5 Rows:")
print(df.head(), "\n")

print("Data Types of All Columns:")
print(df.dtypes, "\n")

# 3. Plot a histogram of the target variable G3
plt.figure(figsize=(8, 5))
# 21 bins cleanly isolate each integer grade value from 0 to 20
plt.hist(df["G3"], bins=21, range=(0, 21), edgecolor="black", color="royalblue", alpha=0.8, align="left")

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.xticks(range(0, 21))
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Save the visualization to the outputs directory
plt.savefig("outputs/g3_distribution.png", bbox_inches="tight")
plt.close()

# --- Exploration Comment ---
# The histogram clearly displays a distinct cluster of zero values sitting 
# completely apart from the main bell-shaped academic distribution. 
# As flagged in the feature guide, these represent situational exam absences 
# rather than an organic grade calculation.

# ==============================================================================
# TASK 2: PREPROCESS THE DATA
# ==============================================================================

print("--- Task 2: Data Preprocessing & Analysis ---")

# 1. Filter out the G3=0 rows
df_filtered = df[df["G3"] > 0].copy()

# Print shapes before and after filtering
print(f"Shape before filtering G3=0: {df.shape}")
print(f"Shape after filtering G3=0 : {df_filtered.shape}")
print(f"Total rows removed: {df.shape[0] - df_filtered.shape[0]}\n")

# --- Reasoning Comment ---
# Keeping the G3=0 rows would distort a linear regression model because these 
# zeros do not reflect low academic ability. They represent a qualitative, 
# situational event (missing the exam). Forcing a straight line to fit both 
# standard performance grades and these extreme zero outliers pulls down the 
# regression plane and ruins the accuracy of its feature weights.

# 2. Convert binary yes/no columns to 1/0
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_filtered[col] = (df_filtered[col] == "yes").astype(int)

# 3. Convert sex column to 0/1 (F=0, M=1)
df_filtered["sex"] = (df_filtered["sex"] == "M").astype(int)

# 4. Compute and print Pearson correlation between absences and G3
corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_filtered["absences"].corr(df_filtered["G3"])

print(f"Correlation between absences and G3 (Original Dataset): {corr_original:.4f}")
print(f"Correlation between absences and G3 (Filtered Dataset) : {corr_filtered:.4f}\n")

# --- Correlation Shift Comment ---
# In the original data, the correlation is very close to zero, making it look 
# like absences do not matter. This happens because the students who missed 
# the exam (G3=0) actually had zero or very few absences leading up to the test. 
# In a scatter plot, these points create a vertical cluster at (0 absences, 0 grade), 
# which completely cancels out the downward trend of the rest of the class. 
# Once filtered, the true negative relationship emerges: more absences clearly 
# correlate with a lower final grade.

# ==============================================================================
# TASK 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================

print("--- Task 3: Exploratory Data Analysis ---")

# 1. Isolate the designated numeric features from the feature guide
numeric_features = [
    "age", "Medu", "Fedu", "traveltime", "studytime", 
    "failures", "absences", "freetime", "goout", "Walc"
]

# Compute correlations with G3 on the filtered dataset
correlations = df_filtered[numeric_features].corrwith(df_filtered["G3"])
correlations_sorted = correlations.sort_values()

print("Pearson correlation with G3 (Sorted from most negative to most positive):")
for feature, score in correlations_sorted.items():
    print(f"  {feature:12}: {score:+.4f}")
print("")

# --- Correlation Insights Comment ---
# The feature with the strongest linear relationship to G3 is "failures" (negative). 
# This makes perfect logical sense, as a history of past academic struggles is a 
# strong signal of systemic learning gaps. 
# What might be surprising is how weak the negative correlation is for weekend alcohol 
# consumption ("Walc") or free time ("freetime") compared to active habits like 
# going out with friends ("goout"). Additionally, mother's education ("Medu") 
# displays a stronger positive relationship with student math outcomes than father's 
# education ("Fedu").

# 2. Visualization 1: Boxplot of G3 by Number of Past Class Failures
plt.figure(figsize=(8, 5))
# Group data manually to avoid external plotting package dependencies
failure_groups = [df_filtered[df_filtered["failures"] == i]["G3"] for i in sorted(df_filtered["failures"].unique())]
plt.boxplot(failure_groups, tick_labels=sorted(df_filtered["failures"].unique()), patch_artist=True,
            boxprops=dict(facecolor="lightcoral", color="darkred"),
            medianprops=dict(color="black", linewidth=1.5))

plt.title("Final Math Grade (G3) Distribution by Past Failures")
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.savefig("outputs/g3_vs_failures_boxplot.png", bbox_inches="tight")
plt.close()

# --- Plot 1 Observation Comment ---
# This boxplot shows a clear downward shift in the median final math grade as the 
# number of past failures increases. Students with 0 past failures maintain a stable 
# median around 11–12, whereas students with 1, 2, or 3 failures see their scores 
# compress heavily toward the passing line, validating it as a critical negative predictor.

# 3. Visualization 2: Boxplot of G3 by Mother's Education Level (Medu)
plt.figure(figsize=(8, 5))
medu_labels = ["0: None", "1: Primary", "2: 5th-9th Grade", "3: Secondary", "4: Higher Ed"]
medu_groups = [df_filtered[df_filtered["Medu"] == i]["G3"] for i in range(5)]
plt.boxplot(medu_groups, tick_labels=medu_labels, patch_artist=True,
            boxprops=dict(facecolor="aquamarine", color="teal"),
            medianprops=dict(color="black", linewidth=1.5))

plt.title("Final Math Grade (G3) Distribution by Mother's Education")
plt.xlabel("Mother's Education Level (Medu)")
plt.ylabel("Final Grade (G3)")
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.savefig("outputs/g3_vs_medu_boxplot.png", bbox_inches="tight")
plt.close()

# --- Plot 2 Observation Comment ---
# This plot highlights that higher maternal educational attainment correlates with 
# an upward shift in student performance distributions. The median score rises 
# steadily from level 1 up to level 4. This matches the positive Pearson correlation 
# value and confirms socio-environmental support as an important academic baseline.


# ==============================================================================
# TASK 4: BASELINE MODEL
# ==============================================================================

print("--- Task 4: Baseline Model (Failures Only) ---")

# 1. Isolate the single baseline feature matrix (X) and target (y)
X_baseline = df_filtered[["failures"]].values
y_baseline = df_filtered["G3"].values

# 2. Split into training and test sets (80/20, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_baseline, y_baseline, test_size=0.20, random_state=42
)

# 3. Fit the LinearRegression model
model_baseline = LinearRegression()
model_baseline.fit(X_train_b, y_train_b)

# 4. Generate predictions and evaluate metrics
y_pred_b = model_baseline.predict(X_test_b)
slope_b = model_baseline.coef_[0]
rmse_b = np.sqrt(mean_squared_error(y_test_b, y_pred_b))
r2_b = r2_score(y_test_b, y_pred_b)

print(f"Baseline Slope (β1): {slope_b:.4f}")
print(f"Baseline Test RMSE : {rmse_b:.4f}")
print(f"Baseline Test R²   : {r2_b:.4f}\n")

# --- Baseline Interpretation Comment ---
# The slope tells us that each past class failure drops a student's expected 
# final math grade by roughly 2.3 points on the 0–20 scale. 
# The Test RMSE shows that our predictions are off by about 3.1 grade points 
# on average, which is quite a wide error margin for a 20-point scale.
# The R² score indicates that past failures account for around 13-14% of the 
# variance in the test set grades. This matches our exploratory data analysis: 
# while "failures" had the strongest individual correlation, no single feature 
# can capture the full complexity of academic outcomes on its own.

# ==============================================================================
# TASK 5: BUILD THE FULL MODEL
# ==============================================================================

print("--- Task 5: Full Regression Model Analysis ---")

# 1. Define the complete list of feature columns from the prompt
feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", 
    "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"
]

# Extract feature matrix X and target vector y from our filtered DataFrame
X_full = df_filtered[feature_cols].values
y_full = df_filtered["G3"].values

# 2. Split into training and test sets (80/20, random_state=42)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42
)

# 3. Fit the full LinearRegression model
model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

# 4. Generate predictions and calculate metrics
y_train_pred_f = model_full.predict(X_train_f)
y_test_pred_f = model_full.predict(X_test_f)

r2_train_f = r2_score(y_train_f, y_train_pred_f)
r2_test_f = r2_score(y_test_f, y_test_pred_f)
rmse_test_f = np.sqrt(mean_squared_error(y_test_f, y_test_pred_f))

print(f"Full Model Train R² : {r2_train_f:.4f}")
print(f"Full Model Test R²  : {r2_test_f:.4f}")
print(f"Full Model Test RMSE: {rmse_test_f:.4f} grade points\n")

# --- Model Comparison Comment ---
# The test R² improved from ~0.134 (Task 4 baseline) to ~0.204 in this full model. 
# While adding more lifestyle features helped explain an extra ~7% of the variance 
# in student grades, the baseline "failures" feature remains the overwhelming driver 
# of predictive power.

# 5. Print feature names paired with coefficients
print("Full Model Feature Coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"  {name:12s}: {coef:+.3f}")
print("")

# --- Interpretation and Surprising Results Comment ---
# Surprising Result #1: "Fedu" (Father's education) has a negative coefficient (-0.117), 
# while "Medu" (Mother's education) has a strong positive coefficient (+0.449). This is 
# unexpected since education generally acts as a positive academic indicator. However, 
# because Medu and Fedu are highly collinear (highly educated people marry each other), 
# the model gives almost all positive weight to Medu, leaving Fedu with a slight negative 
# adjustment artifact.
#
# Surprising Result #2: "schoolsup" (extra educational support) is deeply negative (-1.442). 
# This does not mean tutoring makes students worse. Instead, it reflects a classic selection 
# bias in social data: students are assigned remedial support precisely because they were 
# already struggling heavily in class.

# --- Overfitting & Generalization Comment ---
# The train R² (0.199) and test R² (0.204) are incredibly close. This indicates 
# that the model does not suffer from overfitting. It generalizes cleanly to 
# unseen test data because we are using a simple, linear model with low-capacity 
# features relative to our sample size.

# --- Production Deployment Deployment Strategy Comment ---
# If deploying this model to production to flag students who need early support, 
# I would make the following adjustments to the feature set:
#
# 1. KEEP: "failures", "Medu", "studytime", and "higher". These features show 
#    strong, statistically logical numeric coefficients. They directly reflect a 
#    student's personal academic history and support structure.
# 2. DROP: "Fedu" due to multi-collinear interference with Medu.
# 3. DROP: "schoolsup" because it functions as an administrative flag for existing 
#    failure risk rather than an independent predictive feature.
# 4. DROP: "activities", "internet", and "freetime" because their weights are 
#    extremely close to zero, meaning they add noise without providing any real 
#    predictive value.



# ==============================================================================
# TASK 6: EVALUATE AND SUMMARIZE
# ==============================================================================

# 1. Generate the Predicted vs. Actual Performance Scatter Plot
plt.figure(figsize=(7, 7))
plt.scatter(y_test_pred_f, y_test_f, alpha=0.7, color="teal", edgecolors="black", label="Test Student")

# Determine range limits to ensure a perfectly proportional square canvas
min_val = min(y_test_f.min(), y_test_pred_f.min()) - 1
max_val = max(y_test_f.max(), y_test_pred_f.max()) + 1

# Plot the 45-degree diagonal perfect fit line (y_predicted = y_actual)
plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2, label="Perfect Fit")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Math Grade (ŷ)")
plt.ylabel("Actual Final Math Grade (y)")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

# Save the diagnostic graphic
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.close()

# --- Plot Evaluation Comment ---
# The model struggles heavily at both the extreme low end and high end. Notice 
# how the predictions (x-axis) are tightly compressed between ~9 and ~13, never 
# predicting a very low or very high score. This is a common limitation of linear 
# regression on weak behavioral data—it playing it safe by predicting close to the 
# mean.
# * A point ABOVE the diagonal means the actual grade was higher than predicted (the 
#   model underestimated the student).
# * A point BELOW the diagonal means the actual grade was lower than predicted (the 
#   model overestimated the student).

# ==============================================================================
#                      PLAIN-LANGUAGE MODEL SUMMARY
# ==============================================================================
#
# 1. DATASET SCALE:
#    * Filtered Dataset Size: 357 students (after dropping exam absences).
#    * Evaluation Test Set: 72 students (20% holdout split).
#
# 2. MODEL METRICS & PREDICTION ERROR:
#    * Test R² Score: 0.2036. This indicates that lifestyle, demographic, and 
#      behavioral features explain roughly 20.4% of the variance in final grades. 
#    * Test RMSE: 2.94 grade points. On a standard 0–20 grading scale, a typical 
#      prediction error is about 3 points. For example, if a student's actual 
#      aptitude warrants a grade of 12, the model might easily guess anywhere 
#      between a 9 or a 15, which is quite a wide range for academic placement.
#
# 3. DOMINANT FEATURES & IMPACT:
#    * Largest Positive Coefficient: "higher" (+1.258)
#      Interpretation: Students who actively plan to pursue higher education score 
#      about 1.26 points higher on average. This highlights internal motivation 
#      and long-term ambition as top academic drivers.
#    * Largest Negative Coefficient: "schoolsup" (-1.442)
#      Interpretation: Receiving extra school support is associated with a 1.44 point 
#      drop in final grades. This does not mean tutoring harms performance; it shows 
#      selection bias, as only structurally struggling students are assigned remediation.
#
# 4. SURPRISING RESULT:
#    * "failures" vs. "studytime": While we assume study hours dictate test scores, 
#      "studytime" has a tiny positive weight (+0.329), while past class "failures" 
#      has a massive negative pull (-1.135). Overcoming historical academic deficits 
#      impacts final math scores far more than simply logging extra study hours.
#
# ==============================================================================
print("Task 6 Complete: Performance plot exported to outputs/predicted_vs_actual.png")


# ==============================================================================
# NEGLECTED FEATURE: THE POWER OF G1
# ==============================================================================

print("--- Neglected Feature Analysis: Incorporating G1 ---")

# 1. Expand feature columns to include G1 alongside all behavioral columns
feature_cols_with_g1 = feature_cols + ["G1"]

# 2. Extract feature matrix X and target vector y from our filtered DataFrame
X_g1 = df_filtered[feature_cols_with_g1].values
y_g1 = df_filtered["G3"].values

# 3. Maintain the identical train/test split parameters (80/20, random_state=42)
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.20, random_state=42
)

# 4. Fit the expanded model
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

# 5. Generate predictions and evaluate metrics
y_pred_g1 = model_g1.predict(X_test_g1)
r2_test_g1 = r2_score(y_test_g1, y_pred_g1)
rmse_test_g1 = np.sqrt(mean_squared_error(y_test_g1, y_pred_g1))

print(f"Model with G1 Included - Test R²  : {r2_test_g1:.4f}")
print(f"Model with G1 Included - Test RMSE: {rmse_test_g1:.4f} grade points\n")

# ==============================================================================
#                      G1 INTERPRETATION & REFLECTION
# ==============================================================================
#
# 1. DOES A HIGH R² MEAN G1 CAUSES G3?
#    No, G1 does not "cause" G3. Correlation does not equal causation. G1 is 
#    simply an early measurement of the exact same underlying construct: a 
#    student's mathematical capability, study habits, and grasp of the course 
#    material. G1 acts as a proxy or "sneak peek" for G3 because math curricula 
#    are highly cumulative.
#
# 2. IS THIS A USEFUL MODEL FOR IDENTIFYING STUDENTS WHO MIGHT STRUGGLE?
#    It is highly accurate, but its practical usefulness is limited by timing. 
#    By the time the G1 exam is graded and processed, the first grading period 
#    is already over. Waiting until G1 is available means students have already 
#    spent months falling behind on basic concepts, making intervention harder.
#
# 3. WHAT MUST EDUCATORS DO TO INTERVENE EARLY (BEFORE G1 IS AVAILABLE)?
#    To flag students on day one, educators cannot rely on high-accuracy test 
#    proxies. They must rely on the weaker behavioral and lifestyle features 
#    modeled in Task 5. 
#    * They should build intake risk profiles using background traits (e.g., 
#      flagging students with 1 or more past class "failures").
#    * They should track early diagnostic signals that show up weeks before an 
#      official exam, such as real-time homework completion rates, attendance 
#      and school "absences" within the first 14 days, or low "studytime" flags.
#
# ==============================================================================
print("Analysis Complete: Final scripts are ready for staging and PR submission.")

