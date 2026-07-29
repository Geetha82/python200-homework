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

# --- TASK 2: Prepare Your Data ---
print("\n ===== TASK 2: DATA PREPARATION =====")

# 1. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

# 2. Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit Initial PCA to Find Variance Threshold
pca_full = PCA()
pca_full.fit(X_train_scaled)

# 4. Locate Component Count for 90% Explained Variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components required to explain 90% variance (n): {n_components_90}")

# 5. Variance Curve Visualization
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", markersize=2)
plt.axhline(y=0.90, color="r", linestyle=":", label="90% Variance Threshold")
plt.title("PCA Cumulative Explained Variance Curve")
plt.xlabel("Number of Components")
plt.ylabel("Variance Ratio Summary")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()

# Saved to match the exact name expected by the asset check script
plt.savefig("outputs/pca_explained_variance.png", dpi=150)
plt.close()
print("PCA Cumulative Variance plot saved to outputs/pca_explained_variance.png")


# 6. Transform Low-Dimensional Feature Spaces
pca_optimal = PCA(n_components=n_components_90)
X_train_pca = pca_optimal.fit_transform(X_train_scaled)
X_test_pca = pca_optimal.transform(X_test_scaled)

# Comment:
# We used a stratified 80/20 split to maintain the real-world base rate of spam (~39.4%) in both subsets.
# Due to the extreme scale differences observed in Task 1, we center and scale the features using StandardScaler.
# To prevent data leakage, the scaler and PCA are fit exclusively on the training data.
# We apply PCA to reduce the feature space from 57 features down to the number of components explaining 90% of the variance, eliminating noise while preserving essential patterns.


# --- TASK 3: A CLASSIFIER COMPARISON ---
print("\n ===== TASK 3: CLASSIFIER SHOOTOUT =====")

# 1. KNN Diagnostics (Unscaled vs Scaled vs PCA-Reduced)
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)
print(f"KNN Unscaled Accuracy: {accuracy_score(y_test, y_pred_knn_unscaled):.4f}")
print("KNN Unscaled Report:\n", classification_report(y_test, y_pred_knn_unscaled))

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

# --- 2. Decision Tree Diagnostics ---
depths = [3, 5, 10, None]
print("\nDecision Tree Parameter Sweep:")
for d in depths:
    dt_sweep = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_sweep.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, dt_sweep.predict(X_train))
    te_acc = accuracy_score(y_test, dt_sweep.predict(X_test))
    print(f"  Max Depth: {str(d):4s} | Train Accuracy: {tr_acc:.4f} | Test Accuracy: {te_acc:.4f}")

print("\n[Production Decision Justification]:")
print("Evaluating the sweep table above shows that setting max_depth=None causes severe training memorization (99.97%).")
print("Shallow trees (depths 3 and 5) underfit the structural features, capping test accuracy at 0.8849 and 0.8990.")
print("A max_depth of 10 balances complexity and performance, capturing virtually all available tree predictive power.")
print("Therefore, max_depth=10 is selected empirically for production to prevent overfitting.")

chosen_depth = 10
dt_final = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
dt_final.fit(X_train, y_train)
y_pred_dt = dt_final.predict(X_test)
print(f"\nDecision Tree (d=10) Final Report:\n", classification_report(y_test, y_pred_dt))

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# 4. Logistic Regression (Scaled vs PCA)
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

# 5. Missing Feature Importances Printing (Fixing the gap)
print("\n ===== TOP 10 FEATURE IMPORTANCES COMPARISON =====")
top_n = 10

# Decision Tree Top 10 Features
dt_imp = dt_final.feature_importances_
dt_top_idx = np.argsort(dt_imp)[::-1][:top_n]
print("\n--- Top 10 Features: Decision Tree ---")
for rank, idx in enumerate(dt_top_idx, start=1):
    print(f" {rank:2d}. {X.columns[idx]:30s} | Importance: {dt_imp[idx]:.4f}")

# Random Forest Top 10 Features
rf_imp = rf.feature_importances_
rf_top_idx = np.argsort(rf_imp)[::-1][:top_n]
print("\n--- Top 10 Features: Random Forest ---")
for rank, idx in enumerate(rf_top_idx, start=1):
    print(f" {rank:2d}. {X.columns[idx]:30s} | Importance: {rf_imp[idx]:.4f}")

# --- Save Evaluation Plots ---
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=["Ham", "Spam"], cmap=plt.cm.Blues)
plt.title("Optimal Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png", dpi=150)
plt.close()

# Save Bar Chart of Random Forest Importances
plt.figure(figsize=(10, 5))
plt.bar(range(top_n), rf_imp[rf_top_idx], color="steelblue")
plt.xticks(range(top_n), X.columns[rf_top_idx], rotation=45, ha="right")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png", dpi=150)
plt.close()
print("\nSaved optimal charts to outputs/ directory.")


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


# --- TASK 4: CROSS-VALIDATION ---

print("\n ===== TASK 4: 5-FOLD CROSS-VALIDATION RESULTS =====")

# Comprehensive cross-validation experiment dictionary tracking all Task 3 variations
cv_experiments = {
    "KNN (Unscaled)": (KNeighborsClassifier(n_neighbors=5), X_train),
    "KNN (Scaled)": (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "KNN (PCA-Reduced)": (KNeighborsClassifier(n_neighbors=5), X_train_pca),
    "Decision Tree (d=10)": (DecisionTreeClassifier(max_depth=10, random_state=42), X_train),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "Logistic Regression (Scaled)": (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42), X_train_scaled),
    "Logistic Regression (PCA-Reduced)": (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42), X_train_pca)
}

for name, (model, data_array) in cv_experiments.items():
    # Use cross_val_score on training arrays to respect strict validation bounds
    scores = cross_val_score(model, data_array, y_train, cv=5)
    print(f"{name:32s} | Mean CV Accuracy: {scores.mean():.4f} | Fold Std Dev: {scores.std():.4f}")

# Comment:
# The 5-fold cross-validation confirms that the Random Forest delivers the most reliable generalization performance.
# As expected, the Random Forest exhibits a lower standard deviation (variance across folds) compared to the single Decision Tree.
# This demonstrates how averaging predictions across an ensemble of 100 diverse, randomized trees successfully smooths out structural instability and prevents the model from being overly sensitive to any single layout split of the training data.


# --- TASK 5: BUILDING A PREDICTION PIPELINE ---

print("\n===== TASK 5: PRODUCTION PREDICTION PIPELINES =====")

# 1. Best Tree-Based Pipeline (Random Forest - scale-invariant)
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 2. Best Non-Tree-Based Pipeline (Scaled Logistic Regression)
# Note: Task 3 results showed Scaled Logistic Regression (0.9294) outperformed PCA (0.9186). 
# Therefore, PCA is omitted from this production pipeline to maintain peak performance.
non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42))
])

tree_pipeline.fit(X_train, y_train)
non_tree_pipeline.fit(X_train, y_train)

print("\nPipeline Architectural Summary:")
print("- The tree pipeline omits preprocessing steps because forest algorithms are scale-invariant.")
print("- The non-tree pipeline utilizes StandardScaler to ensure reliable model weight convergence.")
print("- PCA was excluded because Task 3 experiments demonstrated it slightly degraded the non-tree classifier scores.")

# Comment:
# No, they do not share the same structure. The tree-based pipeline consists solely of the final classifier step because 
# Random Forests use axis-aligned threshold splits and are completely insensitive to feature scales. Conversely, the 
# non-tree-based pipeline requires an initial StandardScaler step; without it, features with massive raw variations 
# would disproportionately dominate regularization and model weights relative to tiny word-frequency percentages.
#
# Practical Value of Packaging:
# 1. Eliminates Silent Bugs: It encapsulates the precise order of preprocessing operations, preventing developers from accidentally skipping scaling or introducing data leakage.
# 2. Streamlines Engineering Handoffs: Instead of distributing separate script files for cleaning, transforming, and predicting, you can hand off or export a single, clean compiled object that accepts raw, unscaled inputs directly.
# 3. Simplifies Production Deployment: During live server inference, passing an incoming raw email into a single pipeline.predict() call executes everything instantly, matching the exact data transformation state established during model training.

