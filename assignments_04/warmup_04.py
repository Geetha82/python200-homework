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

# =====================================================================
# --- ROC and AUC ---
# =====================================================================

# --- Q1 ---
print("--- Q1: Baseline ROC-AUC Comparison ---")

# 1. Logistic Regression on raw (unscaled) training data
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Compute predicted probabilities on the test set for the positive class (column 1)
y_probs_lr = lr_model.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_probs_lr)


# 2. K-Neighbors Classifier on scaled training data
# Using a Pipeline ensures training scaling parameters are strictly applied to test data
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)

# Compute predicted probabilities on the test set for the positive class (column 1)
y_probs_knn = knn_pipeline.predict_proba(X_test)[:, 1]
auc_knn = roc_auc_score(y_test, y_probs_knn)


# 3. Print out baseline metrics
print(f"Logistic Regression (Unscaled) Test AUC: {auc_lr:.4f}")
print(f"K-Neighbors Classifier (Scaled) Test AUC: {auc_knn:.4f}")

# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: Which model has a higher AUC?
# A: The K-Neighbors Classifier trained on scaled data has a significantly higher 
#    AUC (0.9394) than the Logistic Regression model trained on unscaled data (0.7060).
#
# Q: What does that tell you about which model better separates the two classes, 
#    independently of any threshold choice?
# A: It indicates that the K-Neighbors Classifier has a much stronger class-separation capacity. 
#    Independently of any specific decision threshold, if we randomly pick one positive 
#    instance and one negative instance from the data, the KNN model has a 93.94% chance 
#    of assigning a higher probability score to the positive instance, whereas the 
#    Logistic Regression model only has a 70.60% chance.


# --- Q2 ---
print("\n--- Q2: ROC Curve Visualization Comparison ---")

# 1. Compute ROC curve arrays for both models
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_probs_knn)

# 2. Initialize the plot
plt.figure(figsize=(8, 6))

# 3. Plot both curves with descriptive labels containing their exact AUC scores
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, 
         label=f'K-Neighbors Classifier (AUC = {auc_knn:.4f})')
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, 
         label=f'Logistic Regression (AUC = {auc_lr:.4f})')

# 4. Add the random-classifier baseline reference diagonal
plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', label='Random Guessing (AUC = 0.5000)')

# 5. Add clean plot styling elements
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Classifier Performance Comparison (ROC Curves)')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)

# 6. Save the visualization artifact
output_image_path = "outputs/roc_comparison.png"
plt.savefig(output_image_path, dpi=300)
plt.close()

print(f"ROC comparison figure successfully compiled and saved to: {output_image_path}")

# =====================================================================
#  ANALYSIS COMMENT:
# =====================================================================
# Q: At the point on each curve where TPR = 0.80, which model has the lower FPR?
# A: The K-Neighbors Classifier (orange curve) has a much lower FPR (~0.06) 
#    compared to the Logistic Regression model (blue curve, FPR ~0.57) at TPR = 0.80.
#
# Q: What does that mean practically — if you needed to catch 80% of positives, 
#    which model would produce fewer false alarms?
# A: Practically, it means the K-Neighbors Classifier will produce significantly 
#    fewer false alarms (false positives). If your business logic dictates that you 
#    must catch 80% of all true positive cases, switching to the KNN model allows 
#    you to meet that goal while misclassifying only about 6% of negative instances, 
#    whereas Logistic Regression would mistakenly flag roughly 57% of negative instances.

# --- Q3 ---
print("\n--- Q3: Threshold Optimization via F1-Score ---")

from sklearn.metrics import f1_score

# 1. Extract thresholds and performance arrays from the logistic regression model
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_probs_lr)

best_f1 = -1.0
best_threshold = None
best_tpr = None
best_fpr = None

# 2. Iterate through all thresholds to calculate operational F1-scores
for idx, thresh in enumerate(thresholds_lr):
    # Apply threshold cutoff mapping rules
    y_pred_at_thresh = (y_probs_lr >= thresh).astype(int)
    current_f1 = f1_score(y_test, y_pred_at_thresh, zero_division=0)
    
    # Track the peak optimization metric
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh
        best_tpr = tpr_lr[idx]
        best_fpr = fpr_lr[idx]

# 3. Print optimal operational metrics
print(f"Optimal Decision Threshold: {best_threshold:.4f}")
print(f"True Positive Rate (TPR / Recall) at Optimum: {best_tpr:.4f}")
print(f"False Positive Rate (FPR) at Optimum: {best_fpr:.4f}")
print(f"Peak Achieved F1-Score: {best_f1:.4f}")

# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: How does this optimal threshold compare to the default 0.5?
# A: The optimal threshold (0.2757) is significantly lower than the default 0.5 cutoff. 
#    By dropping the threshold, we allow the model to predict the positive class more 
#    aggressively, which maximizes our harmonic balance (F1-score) by trading a higher 
#    false alarm rate (FPR = 0.6900) for a much higher capture rate (TPR = 0.8900).
#
# Q: In a real application, when would you choose a threshold lower than 0.5?
# A: You would lower the threshold below 0.5 when missing a positive case (a False Negative) 
#    is significantly more costly or dangerous than triggering a false alarm (a False Positive). 
#    Examples include medical screenings (missing a disease), fraud detection (missing a 
#    stolen card swipe), or severe weather warning systems.


# =====================================================================
# --- GridSearchCV ---
# =====================================================================

print("\n--- GridSearch Question 1: Logistic Regression Tuning ---")

# 1. Build the self-contained pipeline
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Define the hyperparameter grid using the double-underscore step syntax
param_grid_lr = {
    'lr__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# 3. Setup and execute the grid search with 5-fold cross-validation
grid_search_lr = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=param_grid_lr,
    cv=5,
    scoring='roc_auc',
    n_jobs = 1
)
grid_search_lr.fit(X_train, y_train)

# 4. Extract and evaluate performance metrics
best_lr_estimator = grid_search_lr.best_estimator_
y_test_probs_lr = best_lr_estimator.predict_proba(X_test)[:, 1]
test_auc_lr = roc_auc_score(y_test, y_test_probs_lr)

# 5. Print results
print(f"Best C value: {grid_search_lr.best_params_['lr__C']}")
print(f"Best CV AUC score: {grid_search_lr.best_score_:.4f}")
print(f"Test AUC of best estimator: {test_auc_lr:.4f}")


# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: Did the grid search pick the same C you would have guessed by default?
# A: No, the grid search picked C = 100.0, whereas the default parameter 
#    setting in scikit-learn is C = 1.0. This indicates that the dataset 
#    benefits from less regularization (a weaker penalty on coefficient weights).
#
# Q: By how much did the test AUC change compared to the default C=1.0?
# A: The test AUC changed by almost nothing. The unscaled default model in Q1 
#    achieved an AUC of 0.7060, while the tuned, scaled model achieved an AUC 
#    of 0.7057—a microscopic decrease of 0.0003. This indicates that for this 
#    linear model, neither feature scaling nor adjusting the C parameter significantly 
#    improved its class-separation performance.

# --- GridSearch Question 2: Decision Tree Tuning ---
print("\n--- GridSearch Question 2: Decision Tree Tuning ---")

# 1. Build the self-contained pipeline with a Decision Tree Classifier
dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeClassifier(random_state=42))
])

# 2. Define the hyperparameter grid targeting the tree step using double underscores
param_grid_dt = {
    'dt__max_depth': [2, 3, 5, 8, None]
}

# 3. Setup and execute the grid search with 5-fold cross-validation
grid_search_dt = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=param_grid_dt,
    cv=5,
    scoring='roc_auc',
    n_jobs=1
)
grid_search_dt.fit(X_train, y_train)

# 4. Extract and evaluate performance metrics on test data
best_dt_estimator = grid_search_dt.best_estimator_
y_test_probs_dt = best_dt_estimator.predict_proba(X_test)[:, 1]
test_auc_dt = roc_auc_score(y_test, y_test_probs_dt)

# 5. Print results
print(f"Best max_depth value: {grid_search_dt.best_params_['dt__max_depth']}")
print(f"Best CV AUC score: {grid_search_dt.best_score_:.4f}")
print(f"Test AUC of best estimator: {test_auc_dt:.4f}")

# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: Compare the best AUC from Q1 (logistic regression) to this one (decision tree). 
#    Which model would you bring into further development?
# A: The Decision Tree model achieves a significantly higher Test AUC (0.9354) 
#    than the Logistic Regression model (0.7057). Because the Decision Tree has 
#    vastly superior class-separation capabilities on this dataset, I would bring 
#    the Decision Tree forward into further development.
#
# Q: Is AUC the only thing you would consider?
# A: No, AUC is not the only consideration. In a real-world production system, 
#    I would also consider business-specific metrics like Precision and Recall, 
#    the cost of False Positives vs. False Negatives, model interpretability 
#    (Logistic Regression coefficients are easier to explain than complex trees), 
#    computational latency for real-time inference, and the risk of overfitting.

# --- GridSearch Question 3: Extracting Tuning Performance Statistics ---
print("\n--- GridSearch Question 3: CV Results Summary Table ---")

# 1. Extract results dictionary from the Decision Tree grid search
cv_results = grid_search_dt.cv_results_

# 2. Gather parameters, mean scores, and standard deviation scores
means = cv_results['mean_test_score']
stds = cv_results['std_test_score']
params = cv_results['params']

# 3. Zip them together and sort by mean score in descending order (highest AUC first)
sorted_results = sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)

# 4. Print the sorted evaluations
print(f"{'Rank':<5} | {'Mean CV AUC':<12} | {'Std Dev':<8} | {'Parameter Value(s)'}")
print("-" * 60)
for rank, (mean, std, param) in enumerate(sorted_results, 1):
    print(f"#{rank:<4} | {mean:<12.4f} | {std:<8.4f} | {param}")

# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: Find a case where two parameter values have similar mean scores but different standard deviations.
# A: Looking at our results table, max_depth=3 (Mean: 0.9024, Std Dev: 0.0191) and 
#    max_depth=8 (Mean: 0.8811, Std Dev: 0.0257) have relatively close mean scores, 
#    but max_depth=8 has a visibly higher standard deviation.
#
# Q: If you had to choose between them, which would you pick and why?
# A: I would pick max_depth=3. When mean scores are similar, choosing the model with the 
#    lower standard deviation is always preferred because it demonstrates greater stability 
#    and consistency across different data folds. Additionally, a max_depth of 3 creates 
#    a much simpler, more constrained tree than a max_depth of 8, which aligns with the principle 
#    of Occam's Razor—preferring simpler models to lower the risk of overfitting and variance.

# =====================================================================
# --- Model Persistence with joblib ---
# =====================================================================

print("\n--- joblib Question 1: Pipeline Serialization Verification ---")

# 1. Take the best Pipeline from GridSearch Question 1
best_lr_pipe = grid_search_lr.best_estimator_

# 2. Serialize and save the entire pipeline to disk
model_output_path = "models/warmup_model.pkl"
joblib.dump(best_lr_pipe, model_output_path)

# 3. Load the model pipeline back into memory
loaded_clf = joblib.load(model_output_path)

# 4. Generate predictions from both instances for comparison
original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

# 5. Execute structural assertion matrix testing
assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")


# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: What would break if you saved only the logistic regression model (without the scaler) 
#    and then called .predict(X_test) on the loaded model, where X_test is unscaled?
#
# A: If you pass unscaled raw data directly to a bare Logistic Regression model that was 
#    trained on scaled data, the model will run without throwing an explicit syntax error, 
#    but its predictions will be completely broken, corrupted, and untrustworthy. 
#
#    The model calculates its weights and coefficients based strictly on the variance 
#    and mean of the scaled data (where features are centered around 0 with a standard deviation of 1). 
#    If raw features with much larger magnitudes (e.g., thousands or millions) are fed into 
#    those sensitive weights, the mathematical dot product will blow up. The model will produce 
#    garbage probability outputs, severely degrading your model's real-world performance.

# =====================================================================
# --- Simulated prediction script ---
# =====================================================================
print("\n--- joblib Question 2: Isolated Production Inference Simulation ---")

# 1. Load the model pipeline fresh from disk
production_pipeline = joblib.load("models/warmup_model.pkl")

# 2. Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5, 1.2, -0.3, 0.8, 1.0, -0.5, 0.2, 0.9, -1.1, 0.4],
    [-1.0, 0.5, 0.9, -0.7, -0.2, 1.3, -0.8, 0.1, 0.5, -0.3],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

# 3. Generate predictions and probability arrays
# The pipeline automatically scales this raw data using the training set's mean and variance!
sample_predictions = production_pipeline.predict(new_samples)
sample_probabilities = production_pipeline.predict_proba(new_samples)[:, 1]

# 4. Iterate and print out the inference responses cleanly
for idx, (pred, prob) in enumerate(zip(sample_predictions, sample_probabilities), 1):
    print(f"Row {idx} Input Case:")
    print(f"  └─ Predicted Class Label : {pred}")
    print(f"  └─ Positive Class Prob.  : {prob:.4f}")

# =====================================================================
# ANALYSIS COMMENT:
# =====================================================================
# Q: What do you expect the all-zeros row to predict? Why?
#
# A: I expect the all-zeros row to predict Class 1 (or close to it) because it is 
#    highly dependent on the mean behavior captured by our StandardScaler. 
#    
#    Practically, when a raw row containing all zeros is fed into a pipeline, the 
#    scaler shifts those values by subtracting the training set's mean. If the 
#    original training features had a positive mean, a raw input of 0 translates 
#    to a negative z-score after scaling. 
#
#    In this specific execution context, the model mapped those resulting negative 
#    z-scores straight into a confident Class 1 output (Positive Class Probability: 0.6531). 
#    This highlights a critical lesson: a raw vector of zeros does not represent a 
#    "neutral" input to a machine learning model; its meaning is entirely determined by 
#    the scaling transformations learned during training.

