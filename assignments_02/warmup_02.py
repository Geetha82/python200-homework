import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# --- scikit-learn API ---


# Q1
# 1. Prepare data
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 2. Create the model
model = LinearRegression()

# 3. Fit the model
model.fit(years, salary)

# 4. Extract parameters
slope = model.coef_[0]
intercept = model.intercept_

# 5. Predict for 4 and 8 years of experience
test_years = np.array([4, 8]).reshape(-1, 1)
predictions = model.predict(test_years)

# 6. Print labeled results
print("--- Question 1 ---")
print(f"Model Slope (coef_): {slope:.2f}")
print(f"Model Intercept (intercept_): {intercept:.2f}")
print(f"Predicted salary for 4 years of experience: ${predictions[0]:,.2f}")
print(f"Predicted salary for 8 years of experience: ${predictions[1]:,.2f}")
print()


# Q2
# 1. Start with 1D array
x = np.array([10, 20, 30, 40, 50])
print("--- Question 2 ---")
print(f"Original 1D array shape: {x.shape}")

# 2. Reshape to 2D array
X_2d = x.reshape(-1, 1)
print(f"Reshaped 2D array shape: {X_2d.shape}")
print()

# Why scikit-learn needs X to be 2D:
# scikit-learn designed its API to always expect a feature matrix (rows × columns).
# Even if there is only 1 feature, it must be explicitly represented as a single 
# column so the underlying algorithms can treat it consistently alongside datasets 
# that have multiple columns (features).


# Q3
# 1. Generate synthetic dataset with 3 clusters
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# 2. Create and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_clusters)

# 3. Print cluster centers and counts
print("--- Question 3 ---")
print("Cluster Centers:\n", kmeans.cluster_centers_)
counts = np.bincount(labels)
print(f"Points per cluster: Cluster 0: {counts[0]}, Cluster 1: {counts[1]}, Cluster 2: {counts[2]}")
print()

# 4. Generate and customize scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='X', s=200, label='Centers')

plt.title("K-Means Clustering Analysis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# 5. Ensure outputs directory exists and save figure
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/kmeans_clusters.png")
plt.close()


# ==========================================
# --- Linear Regression ---
# ==========================================

# Shared Synthetic Dataset Generation
np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


# Q1
# Create and customize scatter plot
plt.figure(figsize=(8, 5))
scatter = plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.8, edgecolors='k')
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost ($)")

# Add a legend for clarity
cbar = plt.colorbar(scatter, ticks=[0, 1])
cbar.set_ticklabels(["Non-Smoker", "Smoker"])

# Save and close plot to outputs folder
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/cost_vs_age.png")
plt.close()

# Observations:
# Yes, there are two distinct, parallel upward-trending groups visible in the data. 
# The top group consists of smokers, while the bottom group consists of non-smokers.
# This suggests that smoker status shifts the medical costs upward by a massive,
# constant baseline amount, while age increases costs steadily for both groups.


# Q2
# 1. Reshape single feature 'age' to a 2D array X
X_age = age.reshape(-1, 1)

# 2. Perform 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_age, cost, test_size=0.2, random_state=42
)

# 3. Print array shapes
print("--- Linear Regression Question 2 ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print()


# Q3
# 1. Fit the single-feature model
model_age = LinearRegression()
model_age.fit(X_train, y_train)

# 2. Extract slope and intercept
print("--- Linear Regression Question 3 ---")
print(f"Single Feature Slope: {model_age.coef_[0]:.2f}")
print(f"Single Feature Intercept: {model_age.intercept_:.2f}")

# 3. Predict and calculate metrics
y_pred_age = model_age.predict(X_test)
rmse_age = np.sqrt(np.mean((y_pred_age - y_test) ** 2))
r2_age = model_age.score(X_test, y_test)

print(f"Single Feature Test RMSE: {rmse_age:.2f}")
print(f"Single Feature Test R²: {r2_age:.4f}")
print()

# Slope Interpretation:
# The slope indicates that for every 1-year increase in a patient's age, 
# their predicted annual medical cost increases by roughly $256 (ignoring smoker status).


# Q4
# 1. Combine age and smoker into a 2D feature matrix
X_full = np.column_stack([age, smoker])

# 2. Split dataset with identical parameters
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)

# 3. Fit multi-variable model
model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

# 4. Predict and evaluate
y_pred_full = model_full.predict(X_test_f)
r2_full = model_full.score(X_test_f, y_test_f)

print("--- Linear Regression Question 4 ---")
print(f"Single Feature R² (from Q3): {r2_age:.4f}")
print(f"Multi-Feature Test R²: {r2_full:.4f}")

# 5. Extract multi-variable coefficients
print(f"age coefficient: {model_full.coef_[0]:.2f}")
print(f"smoker coefficient: {model_full.coef_[1]:.2f}")
print()

# Smoker Coefficient Interpretation:
# Holding age constant, being a smoker adds an estimated $14,976.24 
# to a patient's predicted annual medical cost compared to a non-smoker.


# Q5
# 1. Create prediction scatter plot using the full model
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_full, y_test_f, alpha=0.8, color="purple", edgecolors='k')

# 2. Add 45-degree diagonal reference line
min_val = min(y_test_f.min(), y_pred_full.min())
max_val = max(y_test_f.max(), y_pred_full.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Model")

# 3. Annotate plots
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost ($)")
plt.ylabel("Actual Cost ($)")
plt.legend()

# 4. Save visualization to outputs folder
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# Plot Error Meanings:
# When a point falls above the red diagonal line, the actual cost was higher than predicted (Underestimation).
# When a point falls below the red diagonal line, the actual cost was lower than predicted (Overestimation).
