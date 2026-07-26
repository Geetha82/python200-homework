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
    classification_report
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

# --- ROC and AUC ---

# Q1
print("\n--- ROC and AUC Q1 Results---")

# 1. Train Logistic Regression on raw unscaled data
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Compute probabilities for the positive class (class 1)
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
log_reg_auc = roc_auc_score(y_test, log_reg_probs)

# 2. Train KNN on scaled training data using a Pipeline
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)

# Compute probabilities for the positive class (class 1)
knn_probs = knn_pipeline.predict_proba(X_test)[:, 1]
knn_auc = roc_auc_score(y_test, knn_probs)

# 3. Print the comparative AUC scores
print(f"Logistic Regression (Unscaled) AUC Score: {log_reg_auc:.4f}")
print(f"K-Neighbors Classifier (Scaled) AUC Score: {knn_auc:.4f}")

# COMMENT: Comparative Analysis & Threshold Independence Takeaway
# The K-Neighbors Classifier (Scaled) achieves a significantly higher AUC score (0.9394) 
# than the unscaled Logistic Regression model (0.7060). 
# 
# This tells us that the KNN model is much better at separating and distinguishing between 
# the two classes. Because AUC is threshold-independent, this performance ranking holds 
# true across all possible classification thresholds. The scaled KNN model will structurally 
# assign a higher probability (or rank score) to a randomly chosen positive instance 
# than to a randomly chosen negative instance far more reliably than the unscaled 
# Logistic Regression model. It also underscores how critical feature scaling is for 
# distance-based algorithms.


# --- Q2 ---
print("\n# ROC and AUC Question 2")

# 1. Compute ROC curve values for both models
fpr_log, tpr_log, _ = roc_curve(y_test, log_reg_probs)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)

# 2. Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# 3. Plot both curves with their respective AUC scores in the labels
plt.plot(fpr_log, tpr_log, color='blue', lw=2, 
         label=f"Logistic Regression (AUC = {log_reg_auc:.4f})")
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, 
         label=f"K-Neighbors Classifier (AUC = {knn_auc:.4f})")

# 4. Add the random-classifier diagonal baseline
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, 
         label="Random Classifier (AUC = 0.50)")

# 5. Format and label the axes
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)

# 6. Save the figure to the requested directory and close the plot
output_path = "outputs/roc_comparison.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"ROC curve comparison plot successfully saved to {output_path}")

# COMMENT: Specific Operating Point Analysis (TPR = 0.80)
# At the point on each curve where TPR = 0.80, the K-Neighbors Classifier has a much lower FPR 
# (approximately 0.06) compared to the Logistic Regression model (approximately 0.58).
# 
# Practically, if your business objective requires catching 80% of all true positive instances,
# the K-Neighbors Classifier is the far superior choice because it minimizes collateral damage.
# It would achieve this target recall while triggering false alarms on only about 6% of the 
# negative population, whereas Logistic Regression would mistakenly flag roughly 58% of negative 
# cases as positive to reach that same 80% target.

# Q3
print("\n--- ROC and AUC Question 3 ---")

from sklearn.metrics import f1_score

# 1. Get fpr, tpr, and thresholds from roc_curve
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, log_reg_probs)

best_f1 = -1
best_thresh = None
best_tpr = None
best_fpr = None

# 2. Iterate through thresholds to find the optimum F1 score
for i, thresh in enumerate(thresholds_log):
    # Skip the arbitrary threshold scikit-learn places at max(y_score) + 1
    if i == 0 and thresh > 1:
        continue
        
    y_pred = (log_reg_probs >= thresh).astype(int)
    current_f1 = f1_score(y_test, y_pred)
    
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_thresh = thresh
        best_tpr = tpr_log[i]
        best_fpr = fpr_log[i]

# 3. Print optimal threshold details
print(f"Optimal Threshold: {best_thresh:.4f}")
print(f"True Positive Rate (TPR) at Optimum: {best_tpr:.4f}")
# Note: This represents the sensitivity or recall at the peak F1 operating point.
print(f"False Positive Rate (FPR) at Optimum: {best_fpr:.4f}")
# Note: This represents the false alarm rate at the peak F1 operating point.
print(f"Maximum F1 Score: {best_f1:.4f}")

# COMMENT: Threshold Comparison & Practical Application Analysis
# The optimal threshold found here (0.2757) is significantly lower than the default 0.5 threshold. 
# Lowering the threshold to 0.2757 flags more samples as positive, boosting the TPR (Recall) 
# to 89.00% and maximizing the F1 balance, even though it causes the False Positive Rate to climb 
# to 69.00%.
# 
# In a real-world application, you would choose a decision threshold lower than 0.5 when the 
# cost or danger of missing a positive instance (a False Negative) is much greater than the 
# cost of dealing with a false alarm (a False Positive). 
# 
# Critical examples include:
# 1. Medical screening: Missing a disease diagnosis is fatal, while a false alarm just leads to extra tests.
# 2. Fraud or threat detection: Missing a hack or stolen card is catastrophic, whereas a false alarm 
#    just prompts a quick verification text.
# 3. Severe weather warnings: Missing an oncoming tornado costs lives, while an unneeded evacuation 
#    is just an inconvenience.


# --- GridSearchCV ---

# Q1
print("\n--- GridSearchCV Questions 1 ---")

# 1. Build the pipeline containing scaling and model steps
gs_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Define the hyperparameter grid using the pipeline prefix format
param_grid = {
    'lr__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# 3. Instantiate and fit GridSearchCV with 5-fold CV using ROC AUC scoring
grid_search = GridSearchCV(
    estimator=gs_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs = 1
)
grid_search.fit(X_train, y_train)

# 4. Compute the final performance metric on the held-out test data
best_model = grid_search.best_estimator_
test_probs = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probs)

# 5. Print the requested parameters and metric outputs
# Clean step parsing to display just the clean numeric value
print(f"Best C value: {grid_search.best_params_['lr__C']}")
print(f"Best CV AUC score: {grid_search.best_score_:.4f}")
print(f"Test AUC of the best estimator: {test_auc:.4f}")

# COMMENT: GridSearch Output Analysis
# The grid search picked a C value of 100.0, which is much higher than the default C=1.0 value 
# scikit-learn assigns by default. A larger C value reduces regularization, allowing the model 
# to fit the training data more aggressively.
# 
# Comparing this to our original unscaled default Logistic Regression model (Test AUC = 0.7060), 
# the test AUC of our optimized, scaled model actually decreased slightly to 0.7057—a trivial 
# drop of 0.0003. This indicates that for this specific dataset and feature space, scaling and 
# tuning regularization parameters did not yield meaningful performance improvements for 
# Logistic Regression, suggesting that the underlying patterns are either linear but noisy, or 
# better suited to non-linear estimators like KNN.

# Q2
print("\n--- GridSearch Questions 2 ---")

# 1. Build a new pipeline replacing Logistic Regression with a Decision Tree
dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeClassifier(random_state=42))
])

# 2. Define the hyperparameter grid for max_depth
param_grid_dt = {
    'dt__max_depth': [2, 3, 5, 8, None]
}

# 3. Instantiate and fit GridSearchCV (n_jobs=1 to avoid VS Code debugger crashes)
grid_search_dt = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=param_grid_dt,
    cv=5,
    scoring="roc_auc",
    n_jobs=1
)
grid_search_dt.fit(X_train, y_train)

# 4. Compute the final performance metric on the held-out test data
best_dt_model = grid_search_dt.best_estimator_
test_probs_dt = best_dt_model.predict_proba(X_test)[:, 1]
test_auc_dt = roc_auc_score(y_test, test_probs_dt)

# 5. Print the requested parameters and metric outputs
print(f"Best max_depth value: {grid_search_dt.best_params_['dt__max_depth']}")
print(f"Best CV AUC score: {grid_search_dt.best_score_:.4f}")
print(f"Test AUC of the best estimator: {test_auc_dt:.4f}")

# COMMENT: Model Comparison and Selection Strategy
# The tuned Decision Tree Classifier achieves a vastly superior Test AUC (0.9354) 
# compared to the best Logistic Regression model (0.7057). Because of this large performance 
# gap, the Decision Tree is clearly the model to bring forward into further development.
# 
# However, AUC is absolutely not the only factor to consider in a real production pipeline. 
# Other vital business and technical considerations include:
# 1. Inference Latency: Decision trees generate predictions extremely quickly, making them 
#    ideal for live APIs, whereas complex ensembles or deep learning models can be slower.
# 2. Interpretability: Decision trees offer clear decision paths, allowing stakeholders to 
#    understand exactly why a sample was flagged (crucial for regulatory compliance).
# 3. Model Size and Overhead: A single shallow tree requires very little storage space and 
#    minimal memory to run, making deployment straightforward.
# 4. Calibration: We must verify if the predicted probabilities map accurately to actual 
#    real-world event frequencies, rather than just ranking them correctly.

# Q3
print("\n--- GridSearch Question 3 ---")

# 1. Extract results dictionary from the Decision Tree grid search (Q2)
cv_results = grid_search_dt.cv_results_

# 2. Extract specific metric arrays and parameters
mean_scores = cv_results['mean_test_score']
std_scores = cv_results['std_test_score']
params = cv_results['params']

# 3. Create a structured, sortable list of dictionaries
results_list = []
for i in range(len(params)):
    results_list.append({
        'param': params[i]['dt__max_depth'],
        'mean': mean_scores[i],
        'std': std_scores[i]
    })

# 4. Sort results by mean performance descending (best to worst)
sorted_results = sorted(results_list, key=lambda x: x['mean'], reverse=True)

# 5. Print out structural performance breakdown
print("Decision Tree max_depth Tuning Results (Best to Worst):")
for res in sorted_results:
    param_display = "None (Unlimited)" if res['param'] is None else f"{res['param']}"
    print(f"max_depth: {param_display:<16} | Mean CV AUC: {res['mean']:.4f} (± Std Dev: {res['std']:.4f})")

# COMMENT: Tuning Robustness and Variance Analysis
# Looking at the results, max_depth=5 (Mean CV AUC: 0.9165) and max_depth=3 (Mean CV AUC: 0.9024) 
# have relatively close mean scores, but different standard deviations (0.0213 vs 0.0191). 
# Another clear case is comparing deeper vs unlimited trees: max_depth=8 (0.8811 ± 0.0257) 
# vs max_depth=None (0.8626 ± 0.0390).
# 
# If choosing between two configurations with similar mean scores, you should pick the one with 
# the lower standard deviation. A lower standard deviation indicates that the model's 
# performance is consistent and stable across different cross-validation folds. A higher 
# standard deviation warns you that the performance is highly sensitive to the specific 
# slice of training data it receives, which means it carries a higher risk of overfitting 
# or acting unpredictably when deployed on real-world data.


# --- Joblib ---

# Q1
print("\n--- joblib Question 1 ---")

# 1. Save the best pipeline from GridSearch Question 1 to disk
model_path = "models/warmup_model.pkl"
joblib.dump(best_model, model_path)

# 2. Load the pipeline back into memory
loaded_clf = joblib.load(model_path)

# 3. Generate predictions using both the original and loaded instances
# Note: best_model is our best_lr_pipe from GridSearch Q1
original_preds = best_model.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

# 4. Enforce strict reproducibility with an assertion statement
assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# COMMENT: Serialization and Preprocessing Pipeline Risk Analysis
# If you only saved the Logistic Regression model without encapsulating it in a Pipeline 
# with the scaler, calling .predict(X_test) on raw, unscaled test data would cause silent, 
# catastrophic failure. 
# 
# The Python script wouldn't throw a hard crash or SyntaxError, but the model would produce 
# highly inaccurate predictions. Because Logistic Regression relies on coefficients optimized 
# specifically for scaled inputs, feeding it raw features (which may have completely different 
# magnitudes or units) causes the mathematical dot product to skew completely. This highlights 
# why saving the complete Pipeline artifact is a mandatory best practice: it guarantees that 
# preprocessing transformations and downstream model inferences always stay perfectly synchronized.


# Q2
print("\n--- joblib Question 2 ---")

# 1. --- Simulated prediction script ---
# Load the model completely fresh from the disk
production_model = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5, 1.2, -0.3, 0.8, 1.0, -0.5, 0.2, 0.9, -1.1, 0.4],
    [-1.0, 0.5, 0.9, -0.7, -0.2, 1.3, -0.8, 0.1, 0.5, -0.3],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

# 2. Generate hard class predictions and soft class probabilities
simulated_preds = production_model.predict(new_samples)
simulated_probs = production_model.predict_proba(new_samples)

# 3. Print out predictions and metrics for each sample row
print("Simulated Real-Time Inference Results:")
for idx, (pred, prob) in enumerate(zip(simulated_preds, simulated_probs)):
    print(f"Row {idx + 1}:")
    print(f"  Predicted Class: {pred}")
    print(f"  Probability Distribution: Class 0 = {prob[0]:.4f}, Class 1 = {prob[1]:.4f}")

# COMMENT: All-Zeros Row Prediction Rationale
# The all-zeros row predicted Class 1 with a probability of 65.31%. 
# 
# This occurs because the decision boundary of a Logistic Regression model is determined by 
# its intercept term, not just its feature coefficients. Mathematically, the probability is 
# calculated using the sigmoid function: 1 / (1 + exp(-(W * X + b))). 
# 
# When the input vector X is entirely zeros, the product (W * X) drops out to exactly 0, leaving 
# only the intercept 'b' (bias term). Since our model's intercept is positive, the sigmoid 
# transformation of that positive bias evaluates to a probability greater than 50% (specifically 
# 65.31% here). This means that in the complete absence of any unique feature evidence, the model's 
# baseline structural bias tilts naturally toward predicting the majority positive class.
