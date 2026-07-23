import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# ----- TASK 1: DATA EXPLORATION ----- 

# 1. Setup directories
os.makedirs("outputs", exist_ok=True)

# 2. Fetch dataset from UCI
print("Fetching Spambase dataset...")
spambase = fetch_ucirepo(id=94)
X = spambase.data.features
y = spambase.data.targets

# Flatten target variable to a clean 1D array to protect models from vector alignment crashes
y_clean = y.values.ravel()

# Combine into a single DataFrame for exploration
df = pd.concat([X, y], axis=1)

# The ucimlrepo library labels the target column as 'Class' (1=Spam, 0=Ham)
target_col = "Class"

# 3. Print exploratory data analysis statistics
total_emails = len(df)
class_counts = df[target_col].value_counts()
class_props = df[target_col].value_counts(normalize=True)

print("\n ===== TASK 1: DATA EXPLORATION RESULTS =====\n")
print(f"Total number of emails: {total_emails}")
print("\nClass Balance Counts:")
print(f" Ham (0): {class_counts[0]}")
print(f" Spam (1): {class_counts[1]}")
print("\nClass Balance Proportions:")
print(f" Ham (0): {class_props[0]:.2%}")
print(f" Spam (1): {class_props[1]:.2%}")
print("\nAccuracy Score Interpretation Note:")
print(f" A baseline model predicting all 'Ham' yields {class_props[0]:.2%} accuracy.")
print(" Raw accuracy must beat this threshold to show any true predictive value.")

# 4. Generate and save boxplots
target_features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feature in target_features:
    plt.figure(figsize=(6, 5))
    
    # Determine if we should apply log scale due to extreme scale/skew
    # capital_run_length_total goes into thousands; log scale makes it readable
    if feature == "capital_run_length_total":
        sns.boxplot(x=target_col, y=feature, data=df)
        plt.yscale("log")
        plt.ylabel(f"{feature} (Log Scale)")
    else:
        sns.boxplot(x=target_col, y=feature, data=df)
        plt.ylabel(feature)
        
    plt.title(f"Distribution of {feature}\nSpam (1) vs Ham (0)")
    plt.xlabel("Email Class")
    
    # Save to outputs directory
    plt.tight_layout()
    output_path = f"outputs/boxplot_{feature}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Successfully generated and saved: {output_path}")

# --- Analytical Comments ---
# Comment 1: The differences between classes are dramatic. Spam emails show significantly higher 
# distributions across all three features (higher occurrences of "free", "!", and capital runs).
# Comment 2: The heavy skew toward zero tells us that the data is highly sparse; most individual keywords 
# or punctuation signatures only show up in a small subset of specific emails.
# Comment 3: Numeric scales vary dramatically because frequencies are tiny fractions (bounded between 0 and 100), 
# while capital letter counts track absolute run totals that scale into the thousands. This scale imbalance 
# will severely bias unscaled distance-based models like KNN or linear models like Logistic Regression.


# =====================================================================
# --- Task 2: Prepare Your Data ---
# =====================================================================
print("\n--- Task 2: Prepare Your Data ---")

# We apply a stratified 80/20 train/test partition to strictly safeguard authentic out-of-sample proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Standardize data to stabilize variance projection vectors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction Strategy
pca_full = PCA()
pca_full.fit(X_train_scaled)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

# Locate the index component marker matching or exceeding the exact 90% target threshold boundary
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
print(f"Number of principal components required to preserve 90% variance: {n_components_90}")

# Plot Cumulative Explained Variance
plt.figure(figsize=(7, 4.5))
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', markersize=2, linestyle='-')
plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Target')
plt.axvline(x=n_components_90, color='g', linestyle='--', label=f'n={n_components_90} Components')
plt.title("Spambase PCA: Cumulative Variance vs. Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png", dpi=150)
plt.close()

# Slice target components to isolate top transformations
X_train_pca = pca_full.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca_full.transform(X_test_scaled)[:, :n_components_90]

# =====================================================================
# --- Task 3: A Classifier Comparison ---
# =====================================================================
print("\n--- Task 3: A Classifier Comparison ---")

# 1. KNN (Unscaled)
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)
print(f"[Model 1] KNN (Unscaled) Test Accuracy: {accuracy_score(y_test, y_pred_knn_unscaled):.4f}")

# 2a. KNN (Scaled)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print(f"[Model 2a] KNN (Scaled) Test Accuracy:   {accuracy_score(y_test, y_pred_knn_scaled):.4f}")

# 2b. KNN (PCA-Reduced)
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print(f"[Model 2b] KNN (PCA) Test Accuracy:      {accuracy_score(y_test, y_pred_knn_pca):.4f}")

# 3. Decision Tree Tuning Diagnostics
depths = [3, 5, 10, None]
print("\nDecision Tree Depth Optimization Trace:")
for d in depths:
    dt_tune = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_tune.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt_tune.predict(X_train))
    test_acc = accuracy_score(y_test, dt_tune.predict(X_test))
    print(f"  Max Depth: {str(d):4s} | Training Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# Production decision parameter designation comment:
# Setting max_depth=None causes extreme overfitting, with training accuracy hitting 99.9% while 
# test accuracy drops compared to max_depth=10. I select max_depth=10 for production because it 
# achieves the highest test accuracy, balancing depth-based feature rules with out-of-sample generalization.

chosen_depth = 10
best_dt = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
best_dt.fit(X_train, y_train)
y_pred_dt = best_dt.predict(X_test)
print(f"\n[Model 3] Decision Tree (depth={chosen_depth}) Test Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

# 4. Random Forest Classifier Implementation
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"[Model 4] Random Forest Test Accuracy:     {accuracy_score(y_test, y_pred_rf):.4f}")

# 5a. Logistic Regression (Scaled)
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_lr_scaled = lr_scaled.predict(X_test_scaled)
print(f"[Model 5a] Logistic Reg (Scaled) Accuracy: {accuracy_score(y_test, y_pred_lr_scaled):.4f}")

# 5b. Logistic Regression (PCA)
lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
print(f"[Model 5b] Logistic Reg (PCA) Accuracy:    {accuracy_score(y_test, y_pred_lr_pca):.4f}")

# Detailed Reports Generation Block
print("\n" + "="*50 + "\nClassification Reports Overview\n" + "="*50)
print("KNN (Scaled):\n", classification_report(y_test, y_pred_knn_scaled))
