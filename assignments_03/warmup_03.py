import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier


# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1
print("--- Preprocessing Question 1 ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Q2
print("\n--- Preprocessing Question 2 ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Columns mean in X_train_scaled:")
print(X_train_scaled.mean(axis=0))
# We fit the scaler on X_train only to prevent data leakage from the test set into the training phase, preserving a true out-of-sample evaluation.

# --- KNN ---
# Q1
print("\n--- KNN Question 1 ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
print(f"Unscaled KNN Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
print("Classification Report:")
print(classification_report(y_test, knn_preds))

# Q2
print("\n--- KNN Question 2 ---")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
knn_scaled_preds = knn_scaled.predict(X_test_scaled)
print(f"Scaled KNN Accuracy: {accuracy_score(y_test, knn_scaled_preds):.4f}")
# Scaling makes little to no difference here because the raw Iris features already share very similar physical scales and narrow baseline variances.

# Q3
print("\n--- KNN Question 3 ---")
cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=5
)
print(f"Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
# This cross-validation output is more trustworthy than a single train/test split because it averages models over 5 different data permutations, eliminating luck.

# Q4
print("\n--- KNN Question 4 ---")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
for k in k_values:
    knn_tune = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_tune, X_train, y_train, cv=5)
    print(f"k = {k:2d} | Mean CV Accuracy: {scores.mean():.4f}")
# I would choose k = 5 because it achieves the peak mean cross-validation accuracy of 0.9750.
# While it ties with k = 7, k = 5 is the better choice for production deployment because a smaller
# neighborhood requires evaluating fewer data points, minimizing computational overhead and
# memory footprint during inference while preserving maximum generalization capacity.

# --- Classifier Evaluation ---
# Q1
print("\n--- Classifier Evaluation Question 1 ---")
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, knn_preds, display_labels=iris.target_names, cmap=plt.cm.Blues
)
plt.title("KNN (k=5) Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/knn_confusion_matrix.png", dpi=150)
plt.close()
print("Saved outputs/knn_confusion_matrix.png")
# The model most often confuses versicolor and virginica, as they share adjacent and slightly overlapping morphological feature profiles.

# --- Decision Trees ---
# Q1
print("\n--- Decision Trees Question 1 ---")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_preds):.4f}")
print("Classification Report:")
print(classification_report(y_test, dt_preds))
# The Decision Tree accuracy is highly competitive with KNN. Because trees split on single feature thresholds at a time, scaled vs. unscaled data has absolutely no effect on its performance.

# --- Logistic Regression and Regularization ---
# Q1
c_values = [0.01, 1.0, 100.0]
print("\n--- Logistic Regression Regularization Question 1 Results ---")

for c in c_values:
    # Wrap liblinear inside OneVsRestClassifier to make it compatible with 1.5+ scikit-learn
    base_lr = LogisticRegression(C=c, max_iter=1000, solver='liblinear', random_state=42)
    lr_model = OneVsRestClassifier(base_lr)
    
    # Fit the model on the scaled training data
    lr_model.fit(X_train_scaled, y_train)
    
    # Extract coefficients across all internal binary classifiers
    total_coef = np.array([estimator.coef_.flatten() for estimator in lr_model.estimators_])
    coef_magnitude = np.abs(total_coef).sum()
    
    # Display results
    print(f"C value: {c:6.2f} | Total Coefficient Magnitude: {coef_magnitude:.4f}")



# --- PCA ---
# Loading block
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

# Q1
print("\n--- PCA Question 1 ---")
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape: {images.shape}")
fig, axes = plt.subplots(1, 10, figsize=(12, 3))
for i in range(10):
    idx = np.where(y_digits == i)[0][0]
    axes[i].imshow(images[idx], cmap="gray_r")
    axes[i].set_title(f"Digit {i}")
    axes[i].axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_digits.png", dpi=150)
plt.close()
print("Saved outputs/sample_digits.png")

# Q2
print("\n--- PCA Question 2 ---")
pca = PCA()
scores = pca.fit_transform(X_digits)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10
)
plt.colorbar(scatter, label="Digit")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Projection of Handwritten Digits")
plt.tight_layout()
plt.savefig("outputs/pca_2d_projection.png", dpi=150)
plt.close()
print("Saved outputs/pca_2d_projection.png")
# Yes, same-digit images show clean tendencies to form clusters together in this low-dimensional 2D principal projection space.

# Q3
print("\n--- PCA Question 3 ---")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(y=0.80, color="r", linestyle=":", label="80% Threshold")
plt.title("Cumulative Explained Variance Curve")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Ratio")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png", dpi=150)
plt.close()
print("Saved outputs/pca_variance_explained.png")
# Looking directly at the generated graph curve, the cumulative variance line crosses the 0.80 horizontal threshold line at exactly the 13th component index marker. Therefore, we need approximately 13 components to explain 80% of the dataset variance.

# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]
fig, axes = plt.subplots(5, 5, figsize=(10, 11))
print("\n--- PCA Question 4 Results ---")

# --- Row 0: Original images ---
for col_idx in range(5):
    axes[0, col_idx].imshow(images[col_idx], cmap='gray_r')
    axes[0, col_idx].set_xticks([])
    axes[0, col_idx].set_yticks([])
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Original", fontsize=12, fontweight='bold', labelpad=15)

# --- Rows 1-4: Reconstructions with varying n_components ---
for row_idx, n in enumerate(n_values, start=1):
    for col_idx in range(5):
        recon_img = reconstruct_digit(col_idx, scores, pca, n_components=n)
        axes[row_idx, col_idx].imshow(recon_img, cmap='gray_r')
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel(f"n = {n}", fontsize=12, fontweight='bold', labelpad=15)

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png", bbox_inches='tight', dpi=150)
plt.close()
print("outputs/pca_reconstructions.png saved in outputs folder")
# The digits become clearly recognizable at around n = 15 components. This aligns perfectly with our variance curve leveling off point, where a tiny minority of components capture the vast majority of dataset structures.

print("\nAll warmup exercises complete.")
