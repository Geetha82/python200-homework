# --- Pre-preprocessing Observation ---
# Observation: The student_performance_math.csv file uses semicolons (;) as delimiters 
# to separate fields instead of standard commas (,). All text columns are enclosed in 
# double quotes, while numerical targets like G1, G2, and G3 are bare numbers. 
# To load this properly with pandas, we must specify the sep=";" parameter.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Task 1: Load and Explore ---

# Observation: The raw CSV file uses semicolons (';') as delimiters 
# rather than standard commas, so the sep=';' parameter is required.

# 1. Load the dataset
df = pd.read_csv('assignments_02/student_performance_math.csv', sep=';')

print("\n===== Task 1: Load and Explore  =====")
# 2. Explore the dataset structure
print("Dataset Shape:", df.shape)
print("\nFirst Five Rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# 3. Create the outputs directory if it doesn't exist
os.makedirs('assignments_02/outputs', exist_ok=True)

# 4. Plot the histogram of G3 (including the zeros)
plt.figure(figsize=(8, 5))
plt.hist(df['G3'], bins=21, range=(0, 20), edgecolor='black', color='skyblue')

# 5. Customize and save the plot
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.grid(axis='y', alpha=0.75)

# Save the plot to the specified directory
plt.savefig('assignments_02/outputs/g3_distribution.png', dpi=300)
plt.close()

print("\nHistogram successfully saved to outputs/g3_distribution.png")

# --- Task 2: Preprocess the Data ---

print("\n===== Task 2: Preprocess the Data =====")

# 1. Remove rows where G3 is exactly 0
print("--- Row Filtering ---")
print("Shape before filtering:", df.shape)

# Create a new dataset keeping only rows where final grade is greater than 0
df_clean = df[df["G3"] > 0].copy()
print("Shape after filtering:", df_clean.shape)

# Reasoning Comment:
# Keeping the G3=0 rows would distort our model. A grade of zero means the student 
# did not show up for the test, not that they are bad at math. If we keep these rows, 
# the model will get confused trying to predict zeros for students who might otherwise 
# be doing well in class.


# 2. Convert text columns to numbers (1 and 0)
# Convert "yes" to 1 and "no" to 0
df_clean["schoolsup"] = df_clean["schoolsup"].map({"yes": 1, "no": 0})
df_clean["internet"] = df_clean["internet"].map({"yes": 1, "no": 0})
df_clean["higher"] = df_clean["higher"].map({"yes": 1, "no": 0})
df_clean["activities"] = df_clean["activities"].map({"yes": 1, "no": 0})

# Convert "M" to 1 and "F" to 0
df_clean["sex"] = df_clean["sex"].map({"M": 1, "F": 0})

print("\nConverted text columns to numbers successfully!")


# 3. Check correlation between absences and G3
print("\n--- Absences vs G3 Correlation Check ---")
corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Original correlation (with zeros):", corr_original)
print("Filtered correlation (without zeros):", corr_filtered)

# Correlation Comment:
# Filtering changes the result because students who missed the exam (G3=0) had 
# completely different numbers of absences. Some had 0 absences, while others 
# had very high numbers of absences. This wide spread of zero grades across all 
# levels of absences made the relationship look weak. Once we drop the exam absences, 
# we can see a clear trend: more absences directly link to lower final grades.

# --- Task 3: Exploratory Data Analysis ---

print("\n===== Task 3: Exploratory Data Analysis =====")

# 1. Calculate how each column links to the final grade (G3)
print("--- Correlation with Final Grade (G3) ---")

# Calculate correlation numbers for all columns against G3
all_correlations = df_clean.corr(numeric_only=True)["G3"]

# Sort the values from most negative to most positive
sorted_correlations = all_correlations.sort_values()
print(sorted_correlations)

# Feature Analysis Comment:
# The feature with the strongest relationship to G3 is "failures" (around -0.29).
# This means students with a history of failing classes tend to get lower math grades.
# It is surprising that "studytime" has a positive but quite small correlation, meaning
# just tracking hours studied doesn't automatically guarantee a high grade in this data.


# 2. Plot 1: Bar chart showing average grade by number of past failures
plt.figure(figsize=(8, 5))

# Calculate the average G3 score for each failure count group
average_by_failure = df_clean.groupby("failures")["G3"].mean()

# Draw a bar chart using those averages
plt.bar(average_by_failure.index, average_by_failure.values, color="coral", edgecolor="black")

plt.title("Average Final Grade by Past Class Failures")
plt.xlabel("Number of Past Failures")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(average_by_failure.index) # Show exact numbers on x-axis
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Save the first plot
plt.savefig("assignments_02/outputs/g3_by_failures_bar.png")
plt.close()

# Plot 1 Comment:
# This bar chart shows a clear downward staircase. Students with zero past failures 
# have the highest average grades (above 11). For every extra past failure a student 
# has, their average grade drops noticeably, down to under 8 for those with 3 failures.


# 3. Plot 2: Scatter Plot of Final Grade vs School Absences
plt.figure(figsize=(8, 5))

# Draw the scatter dots
plt.scatter(df_clean["absences"], df_clean["G3"], color="purple", alpha=0.6, edgecolor="black")

plt.title("Final Math Grade (G3) vs School Absences")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.grid(True, linestyle="--", alpha=0.5)

# Save the second plot
plt.savefig("assignments_02/outputs/g3_vs_absences_scatter.png")
plt.close()

# Plot 2 Comment:
# This scatter plot shows that the highest-scoring students (grades above 15) almost 
# always have very few absences (under 10). As we look further to the right at students 
# with 20 or more absences, we see that nobody is getting top grades, showing that missing 
# too much school sets a firm ceiling on performance.

# --- Task 4: Baseline Model ---

print("\n===== Task 4: Baseline Model =====")

# 1. Select the single feature and target column
X_base = df_clean[["failures"]]
y_base = df_clean["G3"]

# 2. Split data into 80% training and 20% testing
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y_base, test_size=0.20, random_state=42
)

# 3. Create and train the model
model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)

# 4. Make predictions on the test data
y_pred_b = model_base.predict(X_test_b)

# 5. Calculate and print the results
slope_b = model_base.coef_
rmse_b = np.sqrt(np.mean((y_pred_b - y_test_b) ** 2))
r2_b = model_base.score(X_test_b, y_test_b)

print("--- Baseline Results ---")
print("Slope:", slope_b)
print("RMSE:", rmse_b)
print("R2 score:", r2_b)

# Comment Interpretation:
# The slope is roughly -1.9, meaning each past class failure drops the predicted grade by nearly 2 points.
# The RMSE is around 3.2, which means predictions are usually off by about 3.2 points on the 0-20 scale.
# The R2 score is very low (around 0.08), which is exactly what we expected from the correlation step. 
# One feature alone cannot capture all the reasons why grades vary.

# --- Task 5: Build the Full Model ---

print("\n===== Task 5: Build the Full Model =====")

# 1. Set up the features and target arrays
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

# 2. Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Calculate and print R2 scores and RMSE
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

y_pred = model.predict(X_test)
test_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("--- Full Model Metrics ---")
print("Train R2 score:", train_r2)
print("Test R2 score :", test_r2)
print("Test RMSE     :", test_rmse)

# 5. Print each feature name with its coefficient number
print("\n--- Model Coefficients ---")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Comment Interpretations ---

# Compare to Baseline:
# Adding more features helped. The test R2 score increased compared to the 0.08 
# score from Task 4. Including lifestyle factors helps explain student grades better.

# Surprising Coefficient:
# The negative sign for "schoolsup" (extra school support) is surprising because extra tutoring 
# should help. This happens because extra support is given to students who are *already* struggling 
# in class. The negative number means it acts as a flag for students who face difficulties.

# Train vs Test R2 Comparison:
# The train R2 and test R2 scores are close to each other. This small gap means the model 
# is generalized well and is not overfitting to the training data.

# Production Feature Choices:
# I would KEEP "failures", "higher", and "Medu" because they have the largest impact numbers.
# I would DROP "activities" and "traveltime" because their coefficient numbers are very close 
# to zero, meaning they do not change the predicted grade enough to be worth tracking.

# --- Task 6: Evaluate and Summarize ---
print("\n===== Task 6: Evaluate and Summarize =====")

# 1. Get predictions from the full model
y_pred = model.predict(X_test)

# 2. Start the plot
plt.figure(figsize=(8, 6))

# 3. Create the scatter points
plt.scatter(y_pred, y_test, color="green", alpha=0.7, edgecolor="black")

# 4. Add the ideal diagonal line
line_range = [min(y_test), max(y_test)]
plt.plot(line_range, line_range, color="red", linestyle="--", label="Perfect Model")

# 5. Add titles and labels, then save
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Grade (G3)")
plt.ylabel("Actual Final Grade (G3)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.close()

print("Plot successfully saved to outputs/predicted_vs_actual.png")


# comments ---

# Plot Interpretation:
# The model struggles at both the low end and high end. It pulls low scores up 
# and pushes high scores down, meaning the error is not uniform.
# A point ABOVE the red line means the actual grade was higher than predicted.
# A point BELOW the red line means the actual grade was lower than predicted.

# Project Summary:
# 1. Dataset Size: Filtered dataset has 357 rows. Test set has 72 rows.
# 2. Metrics: The RMSE is about 3.1 points. On a 0-20 scale, this means a typical 
#    prediction is off by around 3 full grade points in either direction.
# 3. Main Features: 
#    - Largest positive is "higher" (adds points for wanting to go to college).
#    - Largest negative is "failures" (subtracts points for past class failures).
# 4. Surprise: "schoolsup" is negative because it acts as a tag for students 
#    who were already struggling before receiving help.

# --- Neglected Feature: The Power of G1 ---

print("\n===== Neglected Feature: The Power of G1 =====")

# 1. Add G1 to our list of features
feature_cols_with_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime", "G1"]
X_g1 = df_clean[feature_cols_with_g1].values
y_g1 = df_clean["G3"].values

# 2. Split into 80% training and 20% testing
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.20, random_state=42
)

# 3. Create and train the new model
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

# 4. Calculate and print the new test R2 score
test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print("--- G1 Model Metrics ---")
print("New Test R2 score with G1:", test_r2_g1)

# ---  Answers and Interpretations ---

# 1. Does G1 cause G3?
# No. A high G1 score does not cause a high G3 score. Instead, G1 is simply 
# an early measurement of the student's existing math ability and study habits, 
# which naturally persist into the final period.

# 2. Is this model useful for identifying struggling students?
# Not really. By the time the G1 grade is recorded, the first school period is already 
# over. Using this model to find struggling students is too late because those students 
# have already spent weeks falling behind in class.

# 3. What should educators do to intervene early?
# Educators must look at the background model features (like past class failures, 
# school attendance records, and educational support needs) on day one. This allows them 
# to flag at-risk students and provide tutoring before the first exam even takes place.
