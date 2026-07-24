import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

# Ensure output directories exist
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
# # --- ROC and AUC ---
# ==========================================

# ROC Question 1
print("# ROC Question 1")

# 1. Scale data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Logistic Regression on raw data
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# 3. Train KNN on scaled data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 4. Compute predicted probabilities for the positive class (column 1)
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]

# 5. Compute and print AUC scores
log_reg_auc = roc_auc_score(y_test, log_reg_probs)
knn_auc = roc_auc_score(y_test, knn_probs)

print(f"Logistic Regression AUC Score: {log_reg_auc:.4f}")
print(f"K-Nearest Neighbors AUC Score:  {knn_auc:.4f}")

# Comment for Q1: 
# K-Nearest Neighbors has the higher AUC score (0.9394 vs 0.7060). 
# This tells us that K-Nearest Neighbors separates the two classes much better 
# than Logistic Regression. Since AUC measures performance across all possible 
# classification thresholds, this higher score confirms that KNN has superior 
# overall discriminative power, independently of any threshold choice.

# ROC Question 2
print("\n# ROC Question 2")

# Compute ROC curve coordinates
fpr_log, tpr_log, _ = roc_curve(y_test, log_reg_probs)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)

# Plot both ROC curves on the same axes
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {log_reg_auc:.4f})", color="royalblue", lw=2)
plt.plot(fpr_knn, tpr_knn, label=f"K-Nearest Neighbors (AUC = {knn_auc:.4f})", color="darkorange", lw=2)

# Add the random-classifier diagonal line
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (AUC = 0.50)")

# Format the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)

# Save to outputs/ directory
plt.savefig("outputs/roc_comparison.png", dpi=300)
plt.close()

print("Saved ROC curve comparison plot to outputs/roc_comparison.png")

# Comment for Q2:
# Looking at the plot at TPR = 0.80:
# - K-Nearest Neighbors has a much lower FPR (around 0.05).
# - Logistic Regression has a much higher FPR (around 0.55).
# 
# Practically, if you need to catch 80% of true positives, K-Nearest Neighbors 
# would produce far fewer false alarms (lower FPR) than Logistic Regression.

# ROC Question 3
print("\n# ROC Question 3")
from sklearn.metrics import f1_score

# Get fpr, tpr, and thresholds from roc_curve using Logistic Regression probabilities
fpr_log, tpr_log, thresholds = roc_curve(y_test, log_reg_probs)

best_f1 = -1
best_thresh = None
best_tpr = None
best_fpr = None

# Iterate through each threshold to find the one maximizing F1 score
for i, threshold in enumerate(thresholds):
    y_pred = (log_reg_probs >= threshold).astype(int)
    current_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_thresh = threshold
        best_tpr = tpr_log[i]
        best_fpr = fpr_log[i]

print(f"Optimal Threshold: {best_thresh:.4f}")
print(f"TPR at Optimum:    {best_tpr:.4f}")
print(f"FPR at Optimum:    {best_fpr:.4f}")
print(f"Best F1 Score:     {best_f1:.4f}")

# Comment for Q3: 
# The optimal threshold (0.2757) is significantly lower than the default 0.5. 
# In a real application, you would choose a threshold lower than 0.5 when the cost of a 
# False Negative (missing a positive case) is much higher than the cost of a False 
# Positive (a false alarm). Examples include medical screenings (missing a disease) 
# or fraud detection (failing to catch a stolen credit card transaction).


# ==========================================
# # --- GridSearchCV ---
# ==========================================
print("\n# GridSearchCV  Question 1")

# 1. Build the pipeline containing a scaler and logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Define the parameter grid (use double underscore syntax for pipeline steps)
param_grid = {
    "lr__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# 3. Setup GridSearchCV with 5-fold cross-validation and ROC AUC scoring
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs = 1
)

# 4. Fit the grid search on the raw training data
grid_search.fit(X_train, y_train)

# 5. Extract results
best_c = grid_search.best_params_["lr__C"]
best_cv_auc = grid_search.best_score_

# 6. Evaluate the best estimator directly on the test set
test_probs = grid_search.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probs)

# 7. Print requested metrics
print(f"Best C value:                {best_c}")
print(f"Best CV AUC score:           {best_cv_auc:.4f}")
print(f"Test AUC of best estimator:  {test_auc:.4f}")

# Comment: 
# The grid search picked C = 100.0, which is different from the default C = 1.0. 
# Compared to the default model from Q1 (Test AUC = 0.7060), the test AUC changed 
# by a tiny amount (0.7057 - 0.7060 = -0.0003), meaning tuning C had almost no 
# practical impact on generalization performance for this dataset.

# GridSearch Question 2
print("\n# GridSearch Question 2")

# 1. Build the pipeline containing a scaler and decision tree classifier
pipeline_dt = Pipeline([
    ("scaler", StandardScaler()),
    ("dt", DecisionTreeClassifier(random_state=42))
])

# 2. Define the parameter grid for max_depth
param_grid_dt = {
    "dt__max_depth": [2, 3, 5, 8, None]
}

# 3. Setup GridSearchCV (keeping n_jobs=1 to avoid debugger warnings)
grid_search_dt = GridSearchCV(
    estimator=pipeline_dt,
    param_grid=param_grid_dt,
    cv=5,
    scoring="roc_auc",
    n_jobs=1
)

# 4. Fit the grid search on the raw training data
grid_search_dt.fit(X_train, y_train)

# 5. Extract results
best_depth = grid_search_dt.best_params_["dt__max_depth"]
best_cv_auc_dt = grid_search_dt.best_score_

# 6. Evaluate the best estimator directly on the test set
test_probs_dt = grid_search_dt.predict_proba(X_test)[:, 1]
test_auc_dt = roc_auc_score(y_test, test_probs_dt)

# 7. Print requested metrics
print(f"Best max_depth:              {best_depth}")
print(f"Best CV AUC score:           {best_cv_auc_dt:.4f}")
print(f"Test AUC of best estimator:  {test_auc_dt:.4f}")

# Comment: 
# The Decision Tree model performed substantially better than Logistic 
# Regression, achieving a Test AUC of 0.9354 compared to Logistic Regression's 
# 0.7057. Based purely on predictive power, the Decision Tree is the clear choice 
# to bring into further development. However, AUC is not the only consideration. 
# In a real project, you must also evaluate model interpretability, training and 
# inference speed, storage footprint, implementation complexity, and how robust 
# the model is to future data drift.

# GridSearch Question 3
print("\n# GridSearch Question 3")

# 1. Extract results dictionary from the Decision Tree grid search
cv_results = grid_search_dt.cv_results_

# 2. Extract values for params, means, and standard deviations
params_list = cv_results["params"]
mean_scores = cv_results["mean_test_score"]
std_scores = cv_results["std_test_score"]

# 3. Zip together, map to human-readable strings, and sort descending by mean score
results_summary = list(zip(params_list, mean_scores, std_scores))
results_summary.sort(key=lambda x: x[1], reverse=True)

# 4. Print the sorted metrics
print("Decision Tree Parameter Ranking (Sorted by Mean CV AUC):")
for params, mean, std in results_summary:
    depth_val = params["dt__max_depth"]
    print(f"max_depth: {str(depth_val):<4} | Mean AUC: {mean:.4f} | Std Dev: {std:.4f}")

# Comment: 
# Comparing max_depth: 5 (Mean AUC: 0.9165, Std Dev: 0.0213) and max_depth: 3 
# (Mean AUC: 0.9024, Std Dev: 0.0191), both show relatively similar mean performance. 
# If choosing between them based purely on variance, max_depth: 3 is slightly more stable 
# across folds due to its lower standard deviation (0.0191). 
# Practically, I would select max_depth: 3 here because it offers comparable accuracy 
# while being a simpler, shallower tree structure that carries a lower risk of overfitting 
# and better generalizability.

# ==========================================
# # --- Joblib (Save/Load) ---
# ==========================================

# joblib Question 1
print("\n# joblib Question 1")

# 1. Extract the best estimator pipeline from GridSearch Question 1
best_lr_pipe = grid_search.best_estimator_

# 2. Save the pipeline to the designated models directory
joblib.dump(best_lr_pipe, "models/warmup_model.pkl")

# 3. Load the model back from disk
loaded_clf = joblib.load("models/warmup_model.pkl")

# 4. Generate predictions from both the original and loaded instances
original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

# 5. Assert equality to confirm integrity of the binary serialization
assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Comment: 
# If you save only the logistic regression model without the pipeline's scaler, 
# calling .predict(X_test) on raw, unscaled test data would cause silent feature misalignment 
# and catastrophic performance degradation. The model's learned weights are strictly 
# dependent on the mean and variance scaling of the training data. Passing raw values 
# directly into those coefficients violates the model assumptions and ruins the predictions, 
# which highlights why wrapping steps into a single Pipeline object is critical.

# joblib Question 2
print("\n# joblib Question 2")

# --- Simulated prediction script ---

# 1. Load the model fresh from disk
deployed_clf = joblib.load("models/warmup_model.pkl")

# 2. Define the three hand-crafted test cases (raw, unscaled data)
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

# 3. Generate predictions and class probabilities
simulated_preds = deployed_clf.predict(new_samples)
simulated_probs = deployed_clf.predict_proba(new_samples)[:, 1]

# 4. Display the results for each row
print("Simulated Production Inference Results:")
for i in range(len(new_samples)):
    print(f"Sample {i+1} | Predicted Class: {simulated_preds[i]} | Probability of Class 1: {simulated_probs[i]:.4f}")

# Comment: 
# For the all-zeros row, we expect a predicted probability near 0.5 (maximum uncertainty). 
# This happens because the Pipeline applies a StandardScaler first; if the training features are 
# centered near 0, an input of 0 transforms into a scaled value near 0 (the average value). 
# When all inputs to a logistic regression model are 0, the prediction relies entirely on the 
# intercept (bias) term. Since the dataset has balanced classes, the intercept is close to 0, 
# leading to a probability close to 0.5 (a coin flip).
