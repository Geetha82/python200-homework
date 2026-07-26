import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# --- ROC and AUC ---
# ==========================================

# # Q1
print("--- ROC and AUC: Q1 ---")
# Scale data manually for KNN configuration
scaler_warmup = StandardScaler()
X_train_scaled = scaler_warmup.fit_transform(X_train)
X_test_scaled = scaler_warmup.transform(X_test)

# Train baseline configurations
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Compute predicted probabilities on the test set
y_probs_lr = lr_model.predict_proba(X_test)[:, 1]
y_probs_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

# Compute AUC scores
auc_lr = roc_auc_score(y_test, y_probs_lr)
auc_knn = roc_auc_score(y_test, y_probs_knn)

print(f"Logistic Regression AUC: {auc_lr:.4f}")
print(f"KNN AUC: {auc_knn:.4f}")

# COMMENT: Which model has higher AUC? What does that tell you about which model 
# better separates the two classes, independently of any threshold choice?
# ANSWER: The KNeighborsClassifier (KNN) model achieves a substantially higher AUC 
# score (0.9394) compared to Logistic Regression (0.7060). This demonstrates that 
# KNN possesses much stronger overall classification and discriminative power on this 
# dataset. It tells us that KNN is far more capable of cleanly separating the underlying 
# feature distributions and ranking a randomly selected positive instance higher than a 
# randomly selected negative instance, completely independent of any specific choice of 
# operational decision threshold.

# # Q2
print("\n--- ROC and AUC: Q2 ---")
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_probs_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.4f})")
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc_knn:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")  # Fixed diagonal baseline
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("outputs/roc_comparison.png")
plt.close()
print("ROC comparison plot saved to outputs/roc_comparison.png")

# COMMENT: At the point on each curve where TPR = 0.80, which model has the lower FPR? 
# What does that mean practically — if you needed to catch 80% of positives, which model would produce fewer false alarms?
# ANSWER: Looking at the horizontal line where TPR = 0.80, the KNN model (orange curve) 
# has a drastically lower FPR (roughly 0.06) compared to the Logistic Regression model 
# (blue curve, roughly 0.58). Practically, if your operational requirements dictate 
# catching 80% of the true positive cases, using the KNN model will produce far fewer 
# false alarms, minimizing operational noise, unnecessary alert fatigue, and triage costs.

# # Q3
print("\n--- ROC and AUC: Q3 ---")
fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)

best_f1 = -1
opt_thresh = 0.5
opt_fpr = 0.0
opt_tpr = 0.0

# Step through thresholds to identify optimum F1
for thresh in thresholds:
    if thresh > 1.0:
        continue
    y_pred_step = (y_probs_lr >= thresh).astype(int)
    score = f1_score(y_test, y_pred_step)
    if score > best_f1:
        best_f1 = score
        opt_thresh = thresh

# Match exact metrics at the chosen optimum index
idx = np.argmin(np.abs(thresholds - opt_thresh))
opt_fpr = fpr[idx]
opt_tpr = tpr[idx]

print(f"Optimal Threshold: {opt_thresh:.4f}")
print(f"TPR at Optimum: {opt_tpr:.4f}")
print(f"FPR at Optimum: {opt_fpr:.4f}")
print(f"F1 at Optimum: {best_f1:.4f}")

# COMMENT: how does this optimal threshold compare to the default 0.5? In a real application, 
# when would you choose a threshold lower than 0.5?
# ANSWER: The optimal threshold found is 0.2757, which is significantly lower than the 
# default 0.5 baseline. This lower threshold yields a high TPR of 0.8900 at the expense of a 
# high FPR of 0.6900 to maximize the overall balanced F1 score (0.6899) for this linear model. 
# In a real-world application, you intentionally choose a threshold below 0.5 whenever the cost 
# or penalty of missing a positive case (a False Negative) is far worse than generating a false 
# alarm (a False Positive). Examples include medical diagnoses (e.g., screening for a critical illness) 
# or fraud detection systems, where it is vital to flag potential positives aggressively.



# GridSearch Q1

# Define the pipeline with scaling and logistic regression
gs_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

# Define the hyperparameter grid (use double underscore for pipeline parameters)
param_grid = {
    "lr__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Set up grid search with 5-fold cross-validation scoring on ROC AUC
grid_search = GridSearchCV(
    estimator=gs_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs = 1
)

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# Evaluate the optimized model on the test data
best_model = grid_search.best_estimator_
test_probs = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probs)

# Print performance metrics
print(f"Best C value: {grid_search.best_params_['lr__C']}")
print(f"Best CV AUC score: {grid_search.best_score_:.4f}")
print(f"Test AUC of best estimator: {test_auc:.4f}")

# --- GridSearchCV ---
# GridSearch Q1 Evaluation Comment
# The grid search selected C = 100.0, which differs from the default value of C = 1.0.
# Compared to the baseline model using default settings (which achieved a test AUC of 0.7060), 
# the test AUC of the best estimator dropped slightly by 0.0003 (from 0.7060 down to 0.7057).
# This negligible change indicates that tuning the regularization parameter C on this specific 
# dataset provides no practical performance improvement for Logistic Regression.

# GridSearch Q2

# Define the pipeline with scaling and a decision tree
dt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("dt", DecisionTreeClassifier(random_state=42))
])

# Define the hyperparameter grid for max_depth
dt_param_grid = {
    "dt__max_depth": [2, 3, 5, 8, None]
}

# Set up grid search with 5-fold cross-validation scoring on ROC AUC
dt_grid_search = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=dt_param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs = 1
)

# Fit the grid search on the training data
dt_grid_search.fit(X_train, y_train)

# Evaluate the optimized decision tree on the test data
best_dt_model = dt_grid_search.best_estimator_
dt_test_probs = best_dt_model.predict_proba(X_test)[:, 1]
dt_test_auc = roc_auc_score(y_test, dt_test_probs)

# Print performance metrics
print(f"Best max_depth value: {dt_grid_search.best_params_['dt__max_depth']}")
print(f"Best Decision Tree CV AUC score: {dt_grid_search.best_score_:.4f}")
print(f"Decision Tree Test AUC of best estimator: {dt_test_auc:.4f}")

# --- GridSearchCV ---
# GridSearch Q2 Evaluation Comment
# The tuned Decision Tree (max_depth=5) achieved a test AUC of 0.9354, dramatically outperforming 
# the best Logistic Regression from Q1, which plateaued at a test AUC of 0.7057.
# Based strictly on classification power, the Decision Tree is the clear choice for further development.
# However, AUC is not the only metric to consider in production. 
# We must also evaluate:
# 1. Model interpretability (Logistic Regression provides direct feature coefficients).
# 2. Execution latency and memory footprint at scale.
# 3. Risk of overfitting (trees can be unstable with small data changes compared to linear models).
# 4. Calibration of predicted probabilities (tree probabilities often require post-processing).

# GridSearch Q3

# Extract the results dictionary from the Decision Tree grid search
cv_results = dt_grid_search.cv_results_

# Extract means, standard deviations, and tested parameter choices
means = cv_results['mean_test_score']
stds = cv_results['std_test_score']
params = cv_results['params']

# Match, bundle, and sort them from highest mean score to lowest
sorted_indices = np.argsort(means)[::-1]

print("Decision Tree CV Performance Results (Sorted from Best to Worst):")
for idx in sorted_indices:
    param_val = params[idx]['dt__max_depth']
    print(f"max_depth: {str(param_val):<5} | Mean CV AUC: {means[idx]:.4f} (± {stds[idx]:.4f})")

# --- GridSearchCV ---
# GridSearch Q3 Model Selection Comment
# Comparing max_depth = 5 (Mean: 0.9165, Std: 0.0213) and max_depth = 3 (Mean: 0.9024, Std: 0.0191):
# While depth 5 offers a slightly higher mean score, depth 3 has a lower standard deviation (0.0191 vs 0.0213).
# If forced to choose between two similarly performing configurations, picking the one with the lower 
# standard deviation (max_depth=3) is generally preferred because it represents a more stable, less 
# volatile model across different splits of data. Additionally, choosing max_depth=3 honors Occam's Razor: 
# it creates a simpler, shallower tree that generalizes better and carries less risk of overfitting.

# --- joblib ---
# joblib Q1

# Save the best logistic regression pipeline from GridSearch Question 1
joblib.dump(best_model, "models/warmup_model.pkl")

# Load the saved pipeline back from the disk
loaded_clf = joblib.load("models/warmup_model.pkl")

# Generate test set predictions using both original and reloaded models
original_preds = best_model.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

# Verify that the array elements match exactly across all test cases
assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# --- joblib ---
# joblib Q1 Evaluation Comment
# If you saved only the inner LogisticRegression model (without the scaler) and called 
# .predict(X_test) on raw, unscaled data, the model's predictions would completely break. 
# The model coefficients were optimized on features scaled to a specific mean and variance. 
# Inputting unscaled data causes features with large raw magnitudes to dominate, distorting 
# the mathematical calculations and leading to incorrect, unpredictable classification results.
# This highlights why using an end-to-end scikit-learn Pipeline is a critical best practice.

# --- Q2 ---
print("\n--- Simulated prediction script ---")

# Load model fresh from disk
production_model = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

preds = production_model.predict(new_samples)
probs = production_model.predict_proba(new_samples)

# Print statements match the exact required output style
print(f"Sample 1 -> Predicted Class: {preds[0]}, Probability of Class 1: {probs[0][1]:.4f}")
print(f"Sample 2 -> Predicted Class: {preds[1]}, Probability of Class 1: {probs[1][1]:.4f}")
print(f"Sample 3 -> Predicted Class: {preds[2]}, Probability of Class 1: {probs[2][1]:.4f}")

# COMMENT:
# What do you expect the all-zeros row to predict? Why?
# ANSWER: The all-zeros row predicts a Class 1 probability of 0.6531. 
# While the synthetic dataset has a balanced baseline intercept (~0.03), 
# the raw input of 0.0 is NOT the average value for every feature in the training set. 
# Because StandardScaler subtracts the actual dataset means (e.g., feature 5 mean is -0.51), 
# passing raw zeros actually injects positive shifted values into the pipeline. 
# When those shifted values multiply against the model's trained weights, 
# it sways the final decision boundary to favor Class 1 at a 65.31% probability.
