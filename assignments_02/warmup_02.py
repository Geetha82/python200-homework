import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Make sure the outputs folder exists
os.makedirs("outputs", exist_ok=True)
# ===== scikit-learn Question 1 =====
# 1. Dataset Setup
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 2. Create and Fit Model
model = LinearRegression()
model.fit(years, salary)

# 3. Predictions
years_to_predict = np.array([4, 8]).reshape(-1, 1)
predictions = model.predict(years_to_predict)

# 4. Print the outputs with labels
print("Slope (model.coef_[0]):", model.coef_[0])
print("Intercept (model.intercept_):", model.intercept_)
print("Prediction for 4 years of experience:", predictions[0])
print("Prediction for 8 years of experience:", predictions[1])

# ===== scikit-learn Question 2 =====
print("\n===== scikit-learn Question 2 Results =====")
# 1. Setup the 1D array
x = np.array([10, 20, 30, 40, 50])
print(f"Original 1D shape: {x.shape}")

# 2. Reshape to a 2D array
X_2d = x.reshape(-1, 1)
print(f"Reshaped 2D shape: {X_2d.shape}")

# Q2 Explanation:
# scikit-learn requires X to be a 2D array because its algorithms are built
# to process multiple samples and multiple features simultaneously.
# The structure must always follow: (Number of Samples, Number of Features).
# Even if there is only 1 feature, the 2D layout preserves this standard.


# ===== scikit-learn Question 3 =====

# Generate the data points
X_clusters, y_true = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# 1. Create the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)

# 2. Fit the model and predict labels at the same time
labels = kmeans.fit_predict(X_clusters)


# 3. Print the results
print("\n===== scikit-learn Question 3 Results =====")
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Points in each cluster:", np.bincount(labels))

# 4. Create the scatter plot
plt.figure()

# Plot the dataset points (colored by their assigned cluster label)
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)

# Plot the center marks as big black X's
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", marker="X", s=200)

# Add titles and labels
plt.title("K-Means Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save the picture to the outputs folder
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

print("Plot saved successfully to outputs/kmeans_clusters.png")





# --- Linear Regression ---
# Setup the data
np.random.seed(42)
num_patients = 100

age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

print("Data generation successful!")
print("Age shape:", age.shape)
print("Smoker shape:", smoker.shape)
print("Cost shape:", cost.shape)

# ===== Linear Regression  Question 1: Visualizing Medical Cost vs Age =====
# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.8, edgecolors="k")

# Add labels, title, and formatting
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost ($)")
plt.grid(True, linestyle="--", alpha=0.5)

# Save the plot
plt.savefig("outputs/cost_vs_age.png")
plt.close()

print("\n===== Linear Regression  Question 1: Visualizing Medical Cost vs Age  Results  =====")

print("Plot successfully saved to outputs/cost_vs_age.png")

# Comment analysis:
# Yes, there are two distinct, parallel upward-sloping groups visible.
# This suggests that smoking status acts as a massive baseline cost offset.
# Smokers (represented by the upper cluster) pay significantly more on average
# than non-smokers across all age groups, while both groups experience a 
# steady increase in cost as they get older.

# ===== Linear Regression  Question 2: Train/Test Split with Age Feature =====

# Reshape age to a 2D array (X) and keep cost as 1D (y)
X_age = age.reshape(-1, 1)
y_cost = cost

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_age, y_cost, test_size=0.20, random_state=42
)

# Print shapes of all four arrays
print("\n===== Linear Regression  Question 2: Train/Test Split with Age Feature Results  =====")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# ===== Linear Regression  Question 3: Fit Model, Predict, and Evaluate =====

# --- Linear Regression Question 3 ---
print("\n--- Linear Regression Question 3 ---")

# Create and fit the single-feature model
model_single = LinearRegression()
model_single.fit(X_train, y_train)

# Predict on the test set
y_pred_single = model_single.predict(X_test)

# Calculate metrics manually using basic formulas
rmse_single = np.sqrt(np.mean((y_pred_single - y_test) ** 2))
r2_single = model_single.score(X_test, y_test)

# Print outputs
print("\n===== Linear Regression  Question 3: Fit Model, Predict, and Evaluate Results  =====")
print("Slope (Coefficient):", model_single.coef_[0])
print("Intercept:", model_single.intercept_)
print("RMSE:", rmse_single)
print("R² on test set:", r2_single)

# Interpretation of the single-feature slope:
# In plain English, the slope means that for every 1-year increase in age, 
# the model estimates that a patient's medical cost increases by roughly $254. 
# However, because this model ignores smoker status, this slope is exaggerated 
# to try and bridge the gap between the two separate patient bands.

# ===== Linear Regression  Question 4: Multiple Linear Regression with Age and Smoker =====

# Stack age and smoker side-by-side into a 2D table
X_full = np.column_stack([age, smoker])

# Split the full feature dataset (80/20)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y_cost, test_size=0.2, random_state=42)

# Create and fit the two-feature model
model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

# Evaluate and print results
print("\n===== Linear Regression  Question 4: Multiple Linear Regression with Age and Smoker Results  =====")
r2_full = model_full.score(X_test_f, y_test_f)
print("R² with both features:", r2_full)
print("age coefficient: ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Does adding the smoker flag help?
# Yes, adding the smoker flag helps immensely! The R² score jumps drastically 
# from around 0.17 to over 0.90, explaining almost all variation in the data.

# Interpretation of the smoker coefficient:
# In practical terms, holding age constant, being a smoker adds an estimated 
# fixed penalty of roughly $15,103 to a patient's annual medical costs.

# ===== Linear Regression  Question 5: Predicted vs Actual Plot =====

# Generate predictions using our two-feature model
y_pred_full = model_full.predict(X_test_f)

plt.figure()
# Plot predicted values vs actual true values
plt.scatter(y_pred_full, y_test_f, color="blue")

# Create a perfect diagonal line from min cost to max cost
min_val = min(min(y_pred_full), min(y_test_f))
max_val = max(max(y_pred_full), max(y_test_f))
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Costs")
plt.ylabel("Actual Medical Costs")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

print("\n===== Linear Regression  Question 5:  Predicted vs Actual Plot Results  =====")
print("Plot saved to outputs/predicted_vs_actual.png")

# Meaning of points relative to the diagonal line:
# If a point falls ABOVE the red diagonal line, it means the actual cost was higher 
# than what the model predicted (the model under-predicted the cost).
# If a point falls BELOW the red diagonal line, it means the actual cost was lower 
# than what the model predicted (the model over-predicted the cost).
