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

# ----- TASK 3: A CLASSIFIER COMPARISON -----
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
print(f"KNN Scaled Accuracy:   {accuracy_score(y_test, y_pred_knn_scaled):.4f}")
print("KNN Scaled Report:\n", classification_report(y_test, y_pred_knn_scaled))

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print(f"KNN PCA Accuracy:      {accuracy_score(y_test, y_pred_knn_pca):.4f}")
print("KNN PCA Report:\n", classification_report(y_test, y_pred_knn_pca))

# 3. Decision Tree Diagnostics
depths = [3, 5, 10, None]
print("\nDecision Tree Parameter Sweep:")
for d in depths:
    dt_sweep = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_sweep.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, dt_sweep.predict(X_train))
    te_acc = accuracy_score(y_test, dt_sweep.predict(X_test))
    print(f" Max Depth: {str(d):4s} | Train Accuracy: {tr_acc:.4f} | Test Accuracy: {te_acc:.4f}")

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

lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42)
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
print(f"Logistic Regression PCA Accuracy:    {accuracy_score(y_test, y_pred_lr_pca):.4f}")

# --- Best Model Summary & Evaluation Plots ---
# The Random Forest is the best-performing model, achieving ~95%+ test accuracy.
# Retaining full scaled vectors beats PCA across both distance and linear models. In spam filtering, 
# individual keyword signals are critical; compressing features with PCA drops structural nuances.
# For a spam filter, False Positives (blocking a legitimate email) are much more damaging than False Negatives 
# (letting spam pass). We must optimize to minimize False Positives.

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

# ----- TASK 5: BUILDING A PREDICTION PIPELINE -----
print("\n ===== TASK 5: PRODUCTION PIPELINES =====")

tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=42))
])

tree_pipeline.fit(X_train, y_train)
print("\nTree Pipeline Output:\n", classification_report(y_test, tree_pipeline.predict(X_test)))

non_tree_pipeline.fit(X_train, y_train)
print("\nNon-Tree Pipeline Output:\n", classification_report(y_test, non_tree_pipeline.predict(X_test)))

# Pipeline architectural overview comment:
# The tree pipeline skips the feature scaling step because tree ensembles split on single features sequentially, 
# making them invariant to scale transformations. Packaging models into pipeline objects ensures that preprocessing parameters 
# are correctly tracked, preventing data leakage during deployment.

print("\nAll mini-project assignment tasks finalized successfully.")
