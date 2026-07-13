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

print("\n--- Task 3: Exploratory Data Analysis ---")

# 1. Isolate strictly numeric features for direct correlation calculation
# We explicitly avoid string-based categories that haven't been engineered yet
numeric_features = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
    'failures', 'absences', 'freetime', 'goout', 'Walc'
]

# Calculate Pearson correlation with our target variable G3
correlations = df_filtered[numeric_features].corrwith(df_filtered['G3'])

# Sort from most negative to most positive
sorted_correlations = correlations.sort_values()

print("Numeric Feature Correlations with G3 (Sorted):")
print(sorted_correlations)
print()

# Identification of strongest relationships:
# - Strongest Negative Relationship: 'failures' (~ -0.29)
# - Strongest Positive Relationship: 'Medu' (~ +0.24) or 'studytime' (~ +0.13)
#
# Surprising Insights:
# 'Walc' (weekend alcohol) and 'goout' have tiny, negligible negative correlations.
# This is unexpected, as intuition tells us heavy partying should crater math grades.
# Instead, structural factors like past academic failures and maternal education
# play a far greater foundational role in final grade outcomes.


# --- 2. Data Visualization ---

# Plot A: Box Plot of G3 grouped by Past Class Failures
# Rationale: 'failures' shows the strongest negative correlation. A box plot helps
# us see the distributional shifts and drops in median grades for each failure increment.
plt.figure(figsize=(7, 5))
df_filtered.boxplot(column='G3', by='failures', grid=False, 
                    patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title("Final Math Grade (G3) Distribution by Past Failures")
plt.suptitle("") # Clear pandas automatic subtitle default
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("outputs/g3_by_failures_box.png")
plt.close()

# Plot A Comment/Observation:
# As seen in 'outputs/g3_by_failures_box.png', there is a clear step-down effect. 
# Students with 0 past failures have a median grade around 11-12. This drops 
# dramatically for students with 1, 2, or 3 failures, highlighting 'failures' 
# as an important behavioral threshold feature for our model.


# Plot B: Bar Plot of Mean G3 by Mother's Education Level (Medu)
# Rationale: 'Medu' has the strongest positive correlation. A bar plot cleanly 
# visualizes whether average scores rise with higher maternal educational attainment.
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

# Plot B Comment/Observation:
# Looking at 'outputs/g3_by_medu_bar.png', student performance scales almost
# linearly with maternal education. Students whose mothers have higher education (4) 
# score significantly higher on average than those whose mothers have zero or primary
# education, making this a powerful positive predictive feature.

print("Task 3 Exploratory plots saved to assignments_02/outputs/")

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
print("\n--- Task 5: Full Multiple Regression Model ---")

# 1. Define the complete list of target feature columns from the guide
feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", 
    "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"
]

# Extract feature matrix and target vector from the preprocessed DataFrame
X_full = df_filtered[feature_cols].values
y_full = df_filtered["G3"].values

# 2. Split into training and test sets using the designated seed mapping
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# 3. Fit the full Multiple Linear Regression model
full_model = LinearRegression()
full_model.fit(X_train_f, y_train_f)

# 4. Generate prediction metrics across splits
train_preds_f = full_model.predict(X_train_f)
test_preds_f = full_model.predict(X_test_f)

train_r2_f = r2_score(y_train_f, train_preds_f)
test_r2_f = r2_score(y_test_f, test_preds_f)
test_rmse_f = root_mean_squared_error(y_test_f, test_preds_f)

print(f"Full Model Train R² Score: {train_r2_f:.4f}")
print(f"Full Model Test R² Score:  {test_r2_f:.4f}")
print(f"Full Model Test RMSE:      {test_rmse_f:.4f}\n")

# 5. Print each feature name matched precisely with its coefficient weight
print("Feature Coefficients:")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
print()

# --- CRITICAL ANALYSIS AND INTERPRETATION COMMENTS ---

# Baseline vs. Full Model Comparison:
# Adding more features helped significantly. The test R² improved from ~0.081 
# in the baseline model to ~0.201 in the full model. This means that expanding 
# our scope to include student background and lifestyle features more than doubles 
# our ability to explain variance in academic outcomes (from 8% to roughly 20%).
#
# Surprising Coefficient Signs & Explanations:
# 1. Fedu (Father's Education) is negative (-0.147). This is surprising because 
#    one would expect higher parental education to universally boost scores. 
#    Explanation: This is a classic symptom of multicollinearity. Because Medu 
#    and Fedu are highly correlated with each other, the model awards the positive 
#    predictive weight to Medu, leaving Fedu to pick up the residual variance, 
#    which turns its tracking sign negative mathematically.
# 2. schoolsup (Extra School Support) is strongly negative (-1.352). This looks 
#    paradoxical—why would tutoring lower grades? Explanation: This is reverse 
#    causality. Students are assigned to remedial tutoring *because* they are 
#    struggling. The negative sign confirms tutoring targets lower-performing students.
#
# Train R² vs. Test R² Variance Gap:
# Train R² (~0.235) is quite close to Test R² (~0.201). Because the metric drop-off 
# is very small, it proves that the model is structurally stable and is NOT heavily 
# overfitting to the training partition. It generalizes reasonably well.
#
# Production Deployment Selection & Justification:
# If deploying this model to flag at-risk students for early academic intervention, 
# I would make the following adjustments to the feature selections:
#
# KEEP:
# - 'failures' and 'schoolsup': Strong numeric signals for academic distress.
# - 'higher': Wanting higher education provides strong intrinsic tracking weight (+1.10).
# - 'Medu': Reliable socioeconomic foundational background anchor (+0.55).
# - 'studytime': Directly actionable behavioral target column (+0.44).
#
# DROP:
# - 'sex': While it has an active tracking coefficient (+0.95), using gender as 
#   a feature in an automated educational deployment introduces deep ethical bias, 
#   violates equity standards, and risks institutionalizing systemic social patterns.
# - 'activities', 'internet', 'traveltime', 'freetime': These weights are all tiny 
#   (close to zero) and add unnecessary variance and complexity for little value.
# - 'Fedu': Dropped to clear up the multicollinearity interference with Medu.


# ===== Task 6: Evaluate and Summarize =====
print("\n--- Task 6: Evaluating Predictions and Generating Summary ---")

# 1. Create the Predicted vs. Actual Diagnostic Scatter Plot
plt.figure(figsize=(7, 7))
plt.scatter(test_preds_f, y_test_f, color='darkorange', edgecolor='k', alpha=0.7, s=50, label='Test Data Points')

# Create a clean diagonal reference line (where Predicted == Actual)
# We find the min and max limits from the data bounds to frame the line perfectly
min_val = min(min(test_preds_f), min(y_test_f))
max_val = max(max(test_preds_f), max(y_test_f))
plt.plot([min_val, max_val], [min_val, max_val], color='navy', linestyle='--', linewidth=2, label='Perfect Prediction Line')

# Format the chart according to specifications
plt.title("Predicted vs Actual (Full Model)", fontsize=14, pad=15)
plt.xlabel("Predicted Final Grade (ŷ)", fontsize=12)
plt.ylabel("True Final Grade (y)", fontsize=12)
plt.xlim(min_val - 0.5, max_val + 0.5)
plt.ylim(min_val - 0.5, max_val + 0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')

# Adjust layout boundaries and save the figure
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

print("Diagnostic plot saved to outputs/predicted_vs_actual.png")

# ==============================================================================
# --- EXECUTIVE SUMMARY AND PLOT ANALYSIS ---
# ==============================================================================
#
# Plot Diagnosis:
# - Directional Errors: A data point ABOVE the diagonal reference line means the 
#   model UNDER-predicted the student's grade (the true grade was higher than 
#   expected). A data point BELOW the diagonal line means the model OVER-predicted 
#   the grade (the true grade fell short of expectations).
#
# - Systematic Struggles: The model struggles heavily at BOTH the high and low ends. 
#   At the low end, it fails to predict very low scores (grades 4-7); instead, it 
#   clusters its predictions tightly between 9 and 13. At the high end, it lacks the 
#   resolution to predict top-tier marks (grades 18-20), truncating predictions near 
#   a ceiling of 14. The error is NOT uniform; it compresses variance toward the mean.
#
# ------------------------------------------------------------------------------
# PLAIN-LANGUAGE EXECUTIVE SUMMARY
# ------------------------------------------------------------------------------
# 1. Dataset Dimensions:
#    The total cleaned dataset (after removing students who missed the exam) consists 
#    of 357 students. The test validation set contains a slice of 72 student records.
#
# 2. Performance Evaluation:
#    Our optimized multiple regression model achieved a Test R² of ~0.201 and a 
#    Root Mean Squared Error (RMSE) of ~2.72. In plain English, this means that 
#    knowing a student's lifestyle patterns allows us to explain roughly 20% of 
#    why math grades differ. A typical prediction error means our model misses 
#    a student's true final grade by an average margin of roughly 2.7 points 
#    on the traditional 0-20 scale.
#
# 3. Primary Feature Drivers:
#    - Largest Positive Coefficient: 'higher' (+1.10). Students explicitly stating 
#      an intrinsic desire to pursue university-level higher education see an expected 
#      boost of over 1.1 grade points compared to peers who do not.
#    - Largest Negative Coefficient: 'schoolsup' (-1.35). Extra institutional support 
#      maps to an expected drop of 1.35 grade points. This indicates reverse-causality 
#      in the data architecture—remedial help is assigned to students who are already 
#      falling behind.
#
# 4. Key Surprising Finding:
#    The biggest surprise was that weekend drinking ('Walc') and going out with 
#    friends ('goout') had almost no linear statistical weight on final grades. 
#    Rather than behavioral social distractions causing academic failure, historical 
#    academic foundations ('failures') and socioeconomic baseline indicators ('Medu') 
#    wield significantly more predictive influence over final math proficiency.
# ==============================================================================

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
