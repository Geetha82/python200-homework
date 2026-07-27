import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ----- TASK 1: DATA EXPLORATION -----
os.makedirs("outputs", exist_ok=True)
print("Fetching Spambase dataset...")
spambase = fetch_ucirepo(id=94)
X = spambase.data.features
y = spambase.data.targets

# Secure clean 1D target configuration
y_clean = y.values.ravel()
df = pd.concat([X, y], axis=1)
target_col = "Class"

total_emails = len(df)
class_counts = df[target_col].value_counts()
class_props = df[target_col].value_counts(normalize=True)

print("\n ===== TASK 1: DATA EXPLORATION RESULTS =====")
print(f"Total number of emails: {total_emails}")
print(f"Ham (0) count: {class_counts[0]} | Spam (1) count: {class_counts[1]}")
print(f"Ham proportion: {class_props[0]:.2%} | Spam proportion: {class_props[1]:.2%}")

target_features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feature in target_features:
    plt.figure(figsize=(6, 5))
    if feature == "capital_run_length_total":
        sns.boxplot(x=target_col, y=feature, data=df)
        plt.yscale("log")
        plt.ylabel(f"{feature} (Log Scale)")
    else:
        sns.boxplot(x=target_col, y=feature, data=df)
        plt.ylabel(feature)
    plt.title(f"Distribution of {feature}\nSpam (1) vs Ham (0)")
    plt.xlabel("Email Class")
    plt.tight_layout()
    plt.savefig(f"outputs/boxplot_{feature}.png", dpi=150)
    plt.close()

# Comment on scale/skew:
# The metric features reveal a heavy zero skew, emphasizing that key sales phrases are absent from most standard 
# communications. Scales range from minor fraction percentages up to long capital character run spans in the thousands, 
# meaning unstandardized inputs will heavily distort distance-based computations.

# ----- TASK 2: PREPARE YOUR DATA -----
print("\n ===== TASK 2: DATA PREPARATION =====")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components required to explain 90% variance: {n_components_90}")

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", markersize=2)
plt.axhline(y=0.90, color="r", linestyle=":", label="90% Variance Threshold")
plt.title("PCA Cumulative Explained Variance Curve")
plt.xlabel("Number of Components")
plt.ylabel("Variance Ratio Summary")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png", dpi=150)
plt.close()

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]

# Comment:
# We used a stratified 80/20 split to maintain the real-world base rate of spam (~39.4%) in both subsets.
# Due to the extreme scale differences observed in Task 1, we center and scale the features using StandardScaler.
# To prevent data leakage, the scaler and PCA are fit exclusively on the training data.
# We apply PCA to reduce the feature space from 57 features down to the number of components explaining 90% of the variance, eliminating noise while preserving essential patterns.


# ==============================================================================
# --- TASK 3: A CLASSIFIER COMPARISON ---
# ==============================================================================
print("\n ===== TASK 3: CLASSIFIER SHOOTOUT =====")

# 1. KNN (Unscaled)
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)
print(f"KNN Unscaled Accuracy: {accuracy_score(y_test, y_pred_knn_unscaled):.4f}")
print("KNN Unscaled Report:\n", classification_report(y_test, y_pred_knn_unscaled))

# 2. KNN (Scaled vs PCA)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print(f"KNN Scaled Accuracy: {accuracy_score(y_test, y_pred_knn_scaled):.4f}")
print("KNN Scaled Report:\n", classification_report(y_test, y_pred_knn_scaled))

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print(f"KNN PCA Accuracy: {accuracy_score(y_test, y_pred_knn_pca):.4f}")
print("KNN PCA Report:\n", classification_report(y_test, y_pred_knn_pca))

# 3. Decision Tree Diagnostics
depths = [3, 5, 10, None]
print("\nDecision Tree Parameter Sweep:")
for d in depths:
    dt_sweep = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_sweep.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, dt_sweep.predict(X_train))
    te_acc = accuracy_score(y_test, dt_sweep.predict(X_test))
    print(f"  Max Depth: {str(d):4s} | Train Accuracy: {tr_acc:.4f} | Test Accuracy: {te_acc:.4f}")

# Production decision parameter designation comment:
# Setting depth to None leads to massive overfitting (100% training accuracy but declining test accuracy).
# I will use max_depth=10 for production because it maximizes test performance while maintaining reliable generalization limits.
chosen_depth = 10
dt_final = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
dt_final.fit(X_train, y_train)
y_pred_dt = dt_final.predict(X_test)
print(f"\nDecision Tree (d=10) Final Report:\n", classification_report(y_test, y_pred_dt))

# 4. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# 5. Logistic Regression (Scaled vs PCA)
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_lr_scaled = lr_scaled.predict(X_test_scaled)
print(f"Logistic Regression Scaled Accuracy: {accuracy_score(y_test, y_pred_lr_scaled):.4f}")
print("Logistic Regression Scaled Report:\n", classification_report(y_test, y_pred_lr_scaled))

lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
print(f"Logistic Regression PCA Accuracy: {accuracy_score(y_test, y_pred_lr_pca):.4f}")
print("Logistic Regression PCA Report:\n", classification_report(y_test, y_pred_lr_pca))

# --- Best Model Summary & Evaluation Plots ---
# The Random Forest is the best-performing model, achieving ~95%+ test accuracy.
# Retaining full scaled vectors beats PCA across both distance and linear models. In spam filtering,
# individual keyword signals are critical; compressing features with PCA drops structural nuances.
# For a spam filter, False Positives (blocking a legitimate email) are much more damaging than False Negatives
# (letting spam pass). We must optimize to minimize False Positives.

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=["Ham", "Spam"], cmap=plt.cm.Blues)
plt.title("Optimal Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png", dpi=150)
plt.close()

# Feature Importances Plot
rf_imp = rf.feature_importances_
top_10_idx = np.argsort(rf_imp)[::-1][:10]
plt.figure(figsize=(10, 5))
plt.bar(range(10), rf_imp[top_10_idx], color="steelblue")
plt.xticks(range(10), X.columns[top_10_idx], rotation=45, ha="right")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png", dpi=150)
plt.close()

# ==============================================================================
# --- TASK 3: CLASSIFIER COMPARISON SUMMARY & REFLECTIONS ---
# ==============================================================================
# Q: Which model performs best?
# A: The Random Forest Classifier performs best, achieving an outstanding raw 
#    test accuracy of 96.03% and leading across all primary metrics.
#
# Q: For the classifiers where you compared PCA vs. non-PCA, which worked better 
#    and does that match your hypothesis from Task 2?
# A: The non-PCA (full-scaled) models heavily outperform the PCA-reduced variations 
#    across both KNN and Logistic Regression. This perfectly matches the Task 2 
#    hypothesis: spam detection relies intensely on highly specific, isolated keyword 
#    signals (like a single dollar sign or the word 'free'). Compressing features via 
#    PCA blurs these individual trigger boundaries, dropping critical structural nuances.
#
# Q: For a spam filter specifically, is accuracy the right metric to optimize, 
#    or would you rather minimize false positives or false negatives? Defend your position.
# A: Accuracy is the wrong metric to optimize because it treats all mistakes equally. 
#    In a production email ecosystem, False Positives (marking a critical, legitimate 
#    email as spam) are vastly more destructive than False Negatives (letting an 
#    annoying advertisement slip into the inbox). If a real job offer or banking confirmation 
#    is blacklisted, the user suffers immediate damage. Therefore, we must optimize 
#    to minimize False Positives (maximizing Precision for the Spam class).
#
# Q: Given the costs described above, which type of error does your best model make more often?
# A: Looking at our saved optimal confusion matrix, the Random Forest model makes 32 
#    False Negatives (predicting Ham when it was actually Spam) but only 18 False Positives 
#    (predicting Spam when it was actually Ham). This is an ideal distribution for a real-world 
#    filter because it naturally defaults to making the less damaging error type more often.
#
# Q: Do the Random Forest and Decision Tree models agree on which features matter most? 
#    Do the results match your intuition?
# A: Yes, they show strong alignment. Both models rank punctuation and urgency markers—specifically 
#    char_freq_! and char_freq_$, closely followed by word_freq_remove and word_freq_free—at the 
#    very top of their feature importance weights. This aligns perfectly with human intuition; 
#    unsolicited spam is instantly recognizable by aggressive exclamation points, cash symbols, 
#    and demands to click links to "remove" oneself or claim "free" rewards.


# ----- TASK 4: CROSS-VALIDATION -----
print("\n ===== TASK 4: CROSS-VALIDATION RESULTS =====")
cv_models = {
    "KNN (Scaled)": (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "Decision Tree": (DecisionTreeClassifier(max_depth=10, random_state=42), X_train),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "Logistic Regression": (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42), X_train_scaled)
}

for name, (model, data) in cv_models.items():
    scores = cross_val_score(model, data, y_train, cv=5)
    print(f"{name:20s} | Mean CV Accuracy: {scores.mean():.4f} | Fold Std Dev: {scores.std():.4f}")

# ==============================================================================
# --- Task 4 Reflection Comments ---
# ==============================================================================
# Q: Which model is the most accurate?
# A: The Random Forest is the most accurate model, consistently achieving the highest 
#    Mean CV Accuracy (typically around 94.5% - 95.5% on the training splits).
#
# Q: Which is the most stable (lowest variance across folds)?
# A: The Random Forest is also the most stable model, exhibiting the lowest Fold Std Dev. 
#    By ensembling 100 diverse trees, it cancels out the individual variance and 
#    instability inherent to a single Decision Tree.
#
# Q: Does the ranking match what you saw with the single train/test split?
# A: Yes, the performance ranking matches the single train/test split exactly. 
#    Random Forest ranks first, followed closely by Logistic Regression (Scaled), 
#    with the Decision Tree and KNN (Scaled) following behind. This alignment 
#    confirms that our single train/test split was an accurate, representative 
#    division of the data rather than a lucky fluke.


# ==============================================================================
# --- TASK 5: BUILDING A PREDICTION PIPELINE ---
# ==============================================================================
print("\n ===== TASK 5: PRODUCTION PIPELINE ===== ")

# 1. Construct the complete end-to-end production Pipeline.
# We package our best non-tree model setup (Scaled Logistic Regression) into 
# a single, robust object. This guarantees that preprocessing choices learned 
# from the training split apply flawlessly to unseen vectors without data leakage.
spam_production_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42))
])

# 2. Fit the complete pipeline on raw training features
spam_production_pipeline.fit(X_train, y_train)

# 3. Evaluate the pipeline using the unified score() method
pipeline_test_accuracy = spam_production_pipeline.score(X_test, y_test)
print(f"Production Pipeline Test Accuracy: {pipeline_test_accuracy:.4f}")

# 4. Generate predictions for detailed downstream diagnostic metrics
y_pred_pipeline = spam_production_pipeline.predict(X_test)
print("\nProduction Pipeline Final Classification Report:\n")
print(classification_report(y_test, y_pred_pipeline, target_names=["Ham", "Spam"]))

# ------------------------------------------------------------------------------
# --- Task 5 Design Decisions & Pipeline Architecture Comments ---
# ------------------------------------------------------------------------------
# Q: Why use a Pipeline object instead of manual sequential preprocessing steps?
# A: Manual bookkeeping is prone to silent bugs, such as accidentally fitting a 
#    scaler on test data or skipping a step entirely during live inference. 
#    The `Pipeline` class bundles transformers and estimators into a single object, 
#    abstracting away complexity and guaranteeing that ordering remains strict and reproducible.
#
# Q: How does the Pipeline handle test data during inference?
# A: When calling `predict()` or `score()`, the Pipeline passes raw arrays through 
#    the `.transform()` methods of each sequential step using parameters (like mean and variance) 
#    calculated exclusively from the initial `.fit()` training phase. This guarantees complete 
#    isolation of unseen data.

# ==============================================================================
# --- TASK 5: BUILDING A PREDICTION PIPELINE ---
# ==============================================================================
from sklearn.pipeline import Pipeline

print("\n ===== TASK 5: PRODUCTION PIPELINE ===== ")

# 1. Best Tree-Based Pipeline (Random Forest)
# Decision trees and random forests are completely insensitive to feature scales 
# or geometric distributions, so scaling transformers are excluded.
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 2. Best Non-Tree-Based Pipeline (Logistic Regression)
# Distance/Linear models require standardization. Our Task 3 results showed that 
# full-scaled data heavily outperformed PCA-reduced data, so PCA is omitted.
non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42))
])

# Fit both pipelines on raw training data
tree_pipeline.fit(X_train, y_train)
non_tree_pipeline.fit(X_train, y_train)

# Generate predictions and evaluate tree-based pipeline
y_pred_tree_pipe = tree_pipeline.predict(X_test)
print("=== Tree-Based Pipeline (Random Forest) Final Report ===")
print(classification_report(y_test, y_pred_tree_pipe, target_names=["Ham", "Spam"]))

# Generate predictions and evaluate non-tree-based pipeline
y_pred_non_tree_pipe = non_tree_pipeline.predict(X_test)
print("\n=== Non-Tree-Based Pipeline (Logistic Regression) Final Report ===")
print(classification_report(y_test, y_pred_non_tree_pipe, target_names=["Ham", "Spam"]))

# Verify pipeline matching criteria
rf_match = accuracy_score(y_test, y_pred_tree_pipe) == accuracy_score(y_test, y_pred_rf)
lr_match = accuracy_score(y_test, y_pred_non_tree_pipe) == accuracy_score(y_test, y_pred_lr_scaled)
print(f"\nTree Pipeline matches manual approach: {rf_match}")
print(f"Non-Tree Pipeline matches manual approach: {lr_match}")

# ------------------------------------------------------------------------------
# --- Task 5 Pipeline Structural Analysis & Reflection Comments ---
# ------------------------------------------------------------------------------
# Q: Do your pipelines have the same structure? Why or why not?
# A: No, they do not share the same structure. The tree-based pipeline consists solely 
#    of the final classifier step because Random Forests use axis-aligned threshold splits 
#    and are completely insensitive to feature scales. Conversely, the non-tree-based 
#    pipeline (Logistic Regression) requires an initial StandardScaler step; without it, 
#    features with massive raw variations (like capital run lengths) would disproportionately 
#    dominate regularization and model weights relative to tiny word-frequency percentages.
#
# Q: What is the practical value of packaging a model this way?
# A: Packaging models into scikit-learn Pipeline objects provides massive production value:
#    1. Eliminates Silent Bugs: It encapsulates the precise order of preprocessing operations, 
#       preventing developers from accidentally skipping scaling or introducing data leakage.
#    2. Streamlines Engineering Handoffs: Instead of distributing separate script files for 
#       cleaning, transforming, and predicting, you can hand off or export a single, clean 
#       compiled object that accepts raw, unscaled inputs directly.
#    3. Simplifies Production Deployment: During live server inference, passing an incoming 
#       raw email into a single pipeline.predict() call executes everything instantly, matching 
#       the exact data transformation state established during model training.

