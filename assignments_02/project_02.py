# ==============================================================================
# Preprocessing Observation:
# The raw 'student_performance_math.csv' file uses semicolons (';') as a delimiter
# rather than commas, with string values wrapped in double quotes. To correctly
# load this into a DataFrame, the 'sep=";"' parameter must be specified in 
# pd.read_csv().
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
# 5. Ensure outputs directory exists and save figure
os.makedirs("outputs", exist_ok=True)
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
# In the original data, the correlation is artificially close to zero, making it look 
# like absences do not predict grades. This happens because the students with G3=0 
# were actually present all semester—accumulating zero or very few absences—but then 
# did not take the final exam. In the raw dataset, these rows create a dense cluster 
# of points at (0 absences, 0 grade). This position completely counteracts and flatlines 
# the natural downward slope of the rest of the class. Once we filter out the students 
# who skipped the exam, the distortion disappears, and the true negative relationship 
# emerges: more absences clearly predict lower performance.

# ==============================================================================
# TASK 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================

print("--- Task 3: Exploratory Data Analysis ---")

# Isolate numeric features specified in the Feature Guide
numeric_features = [
    "age", "Medu", "Fedu", "traveltime", "studytime", 
    "failures", "absences", "freetime", "goout", "Walc"
]

# Compute Pearson correlations with G3 using the filtered dataset
correlations = df_filtered[numeric_features].corrwith(df_filtered["G3"])
correlations_sorted = correlations.sort_values()

print("Pearson correlation with G3 (Sorted from most negative to most positive):")
for feature, score in correlations_sorted.items():
    print(f"  {feature:12}: {score:+.4f}")
print("")

# --- VISUALIZATION SELECTION AND MOTIVATION COMMENT ---
# Based on the sorted correlation results above, my choice of visualizations 
# is directly guided by the strongest indicators in the filtered data:
# 1. "failures" has the strongest negative correlation (-0.2933), making it 
#    the top candidate to visualize as an academic barrier.
# 2. "Medu" (Mother's education) has the strongest positive correlation (+0.2925), 
#    making it the top candidate to visualize as an academic advantage.

# Plot 1: Boxplot of G3 vs. Past Class Failures (Guided by top negative correlation)
plt.figure(figsize=(8, 5))
failure_groups = [df_filtered[df_filtered["failures"] == i]["G3"] for i in sorted(df_filtered["failures"].unique())]
plt.boxplot(failure_groups, tick_labels=sorted(df_filtered["failures"].unique()), patch_artist=True,
            boxprops=dict(facecolor="lightcoral", color="darkred"),
            medianprops=dict(color="black", linewidth=1.5))

plt.title("G3 Grade Distribution by Past Failures (Top Negative Predictor)")
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.savefig("outputs/g3_vs_failures_boxplot.png", bbox_inches="tight")
plt.close()

# --- Plot 1 Observation ---
# This plot visually confirms the strong negative correlation found in our EDA. 
# As the number of past failures increases from 0 to 3, the median final math grade 
# drops sharply. The entire grade distribution compresses downward, validating why 
# "failures" is our most dominant individual negative tracking feature.

# Plot 2: Boxplot of G3 vs. Mother's Education (Guided by top positive correlation)
plt.figure(figsize=(8, 5))
medu_labels = ["0: None", "1: Primary", "2: 5th-9th", "3: Secondary", "4: Higher Ed"]
medu_groups = [df_filtered[df_filtered["Medu"] == i]["G3"] for i in range(5)]
# Change labels=medu_labels to tick_labels=medu_labels
plt.boxplot(medu_groups, tick_labels=medu_labels, patch_artist=True,
            boxprops=dict(facecolor="aquamarine", color="teal"),
            medianprops=dict(color="black", linewidth=1.5))

plt.title("G3 Grade Distribution by Mother's Education (Top Positive Predictor)")
plt.xlabel("Mother's Education Level (Medu)")
plt.ylabel("Final Grade (G3)")
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.savefig("outputs/g3_vs_medu_boxplot.png", bbox_inches="tight")
plt.close()

# --- Plot 2 Observation ---
# This plot visually confirms the strong positive correlation found in our EDA. 
# There is a steady upward trend in student final grades as maternal education 
# levels increase. Students whose mothers achieved higher education (level 4) 
# exhibit a notably higher median and upper quartile score compared to those at level 1.


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

plt.figure(figsize=(7, 7))
plt.scatter(y_test_pred_f, y_test_f, alpha=0.7, color="teal", edgecolors="black", label="Students")
min_val = min(y_test_f.min(), y_test_pred_f.min()) - 1
max_val = max(y_test_f.max(), y_test_pred_f.max()) + 1
plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2, label="Perfect Fit")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Math Grade (ŷ)")
plt.ylabel("Actual Final Math Grade (y)")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.close()

# --- Plot Error Analysis Comment ---
# The model struggles significantly at the extreme low and extreme high ends 
# of the grade spectrum. It compresses its predictions safely between 9 and 13.
# * A value ABOVE the diagonal line means the actual grade was higher than 
#   what the model predicted, meaning the model underestimated that student.
# * A value BELOW the diagonal line means the actual grade was lower than 
#   what the model predicted, meaning the model overestimated that student.

# --- PLAIN-LANGUAGE SUMMARY COMMENT STATEMENTS ---
# The total size of the filtered dataset used for this analysis is 357 students.
# Out of these, the test set holdout used to evaluate model performance contains 72 students.
# The best background model achieved a test R² of 0.2036 and a test RMSE of 2.9431.
# In plain English, an R² of ~0.20 means our background features only capture about 20% 
# of why grades vary, leaving 80% unexplained. An RMSE of ~2.94 means that on a 0-20 scale, 
# a typical prediction is off by about 3 full grade points, which is a wide error margin.
# The feature with the largest positive coefficient is "higher" (+1.258). This means that 
# when controlling for other features, a student who wants to pursue higher education 
# is predicted to score about 1.26 points higher on their final exam.
# The feature with the largest negative coefficient is "schoolsup" (-1.442). This means 
# that students receiving extra educational support score 1.44 points lower on average. 
# One result that surprised me was the deep negative weight of "schoolsup". This is an 
# example of selection bias in data engineering: the tutoring support itself isn't 
# lowering grades, but rather, students are only given tutoring because they were 
# already falling behind.



# ==============================================================================
# NEGLECTED FEATURE: THE POWER OF G1 (EXTENSION REVISION)
# ==============================================================================

print("--- Neglected Feature Analysis: Incorporating G1 ---")

feature_cols_with_g1 = feature_cols + ["G1"]
X_g1 = df_filtered[feature_cols_with_g1].values
y_g1 = df_filtered["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.20, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

y_pred_g1 = model_g1.predict(X_test_g1)
print(f"Model with G1 Included - Test R²  : {r2_score(y_test_g1, y_pred_g1):.4f}")
print(f"Model with G1 Included - Test RMSE: {np.sqrt(mean_squared_error(y_test_g1, y_pred_g1)):.4f}\n")

# --- MULTI-PROMPT EXTENSION DISCUSSION COMMENTS ---
# Prompt 1: Does a high R² here mean G1 is causing G3?
# No, a high R² does not mean that scoring well on the G1 exam causes a student 
# to score well on G3. G1 is simply an early proxy measure of the same underlying 
# academic ability and study patterns. Math is highly cumulative, so performance 
# on early coursework naturally maps strongly to performance on final exams.
#
# Prompt 2: Is this a useful model for identifying students who might struggle?
# While highly predictive, this model is not very useful for proactive intervention. 
# By the time the G1 exam is taken, graded, and fed into an analysis pipeline, 
# several months of the school year have passed. Waiting for a formal grade means 
# identifying struggling students only after they have already fallen behind.
#
# Prompt 3: What might educators need to do if they wanted to intervene early, before G1 is even available?
# To intervene early, educators must bypass late-stage test proxies entirely. They 
# need to use the lower-accuracy behavioral and background features modeled in Task 5. 
# This means flagging students on day one based on historic factors like past "failures", 
# or tracking real-time behavioral signals during the first two weeks of class—such as 
# early homework submission rates, drop-offs in attendance, and initial absences.

