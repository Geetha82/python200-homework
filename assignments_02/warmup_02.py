import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

# 4. Labeled Outputs
print("\n===== scikit-learn Question 1 Results =====")
print(f"Slope     : {model.coef_[0]:.2f}")
print(f"Intercept : {model.intercept_:.2f}")
print(f"Salary prediction for 4 years: ${predictions[0]:,.2f}")
print(f"Salary prediction for 8 years: ${predictions[1]:,.2f}")

# ===== scikit-learn Question 2 =====
print("\n===== scikit-learn Question 2 Results =====")
# 1. Start with the 1D array
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
# 1. Generate Dataset
X_clusters, _ = make_blobs(
    n_samples=120, centers=3, cluster_std=0.8, random_state=7
)

# 2. Create, Fit, and Predict
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_clusters)

# 3.Print the cluster centers 
print("\n===== scikit-learn Question 3 Results  =====")
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

# 4. Plot and Save
os.makedirs("outputs", exist_ok=True)
plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="black",
    marker="x",
    s=100,
)
plt.title("K-Means Clusters")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("outputs/kmeans_clusters.png")
plt.close()


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

# 1. Create and train the model
model_age = LinearRegression()
model_age.fit(X_train, y_train)

# 2. Print the slope and intercept
print("Slope:", model_age.coef_[0])
print("Intercept:", model_age.intercept_)

# 3. Make predictions on the test data
y_pred = model_age.predict(X_test)

# 4. Calculate and print performance metrics
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model_age.score(X_test, y_test)

print("\n===== Linear Regression  Question 3: Fit Model, Predict, and Evaluate Results  =====")
print("RMSE:", rmse)
print("R2 score:", r2)

# Comment Interpretation:
# The slope tells us how much the medical cost changes for each year we age.
# For example, a slope of 240 means that for every 1 year a person gets older, 
# their predicted annual medical cost goes up by about $240.

# ===== Linear Regression  Question 4: Multiple Linear Regression with Age and Smoker =====

# 1. Combine age and smoker into a 2D matrix
X_full = np.column_stack([age, smoker])

# 2. Split into 80% training and 20% testing
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, cost, test_size=0.20, random_state=42
)

# 3. Create and train the new model
model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

print("\n===== Linear Regression  Question 4: Multiple Linear Regression with Age and Smoker Results  =====")
# 4. Print and compare R2 scores
r2_full = model_full.score(X_test_f, y_test_f)
print("New R2 score:", r2_full)
print("Old R2 score:", r2)

# 5. Print both coefficients
print("age coefficient: ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Comment Interpretation:
# The smoker coefficient is the extra cost added for being a smoker.
# It means a smoker is predicted to pay about $15,000 more per year 
# than a non-smoker of the exact same age.
# Adding this flag helps because the R2 score jumped close to 1.0.

# ===== Linear Regression  Question 5: Predicted vs Actual Plot =====

# 1. Get predictions from the two-feature model
y_pred_f = model_full.predict(X_test_f)

# 2. Start the plot
plt.figure(figsize=(8, 6))

# 3. Draw the scatter points (X = predictions, Y = real values)
plt.scatter(y_pred_f, y_test_f, color="blue", alpha=0.7, edgecolors="k")

# 4. Create a diagonal reference line (min value to max value)
line_range = [min(y_test_f), max(y_test_f)]
plt.plot(line_range, line_range, color="red", linestyle="--", label="Perfect Model")

# 5. Add titles, labels, and save
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Cost ($)")
plt.ylabel("Actual Medical Cost ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

print("\n===== Linear Regression  Question 5: Fit Model, Predict, and Evaluate Results  =====")
print("Plot successfully saved to outputs/predicted_vs_actual.png")

# Comment Interpretation:
# When a point falls ABOVE the red diagonal line, it means the actual cost 
# was higher than what the model predicted (the model under-predicted).
# When a point falls BELOW the red line, it means the actual cost was 
# lower than what the model predicted (the model over-predicted).


