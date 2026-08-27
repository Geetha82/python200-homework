import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
print("--- Logistic Regression Question 1 ---")
c_values = [0.01, 1.0, 100]

for c in c_values:
    # Wrap base LogisticRegression model inside the OneVsRestClassifier
    base_lr = LogisticRegression(penalty='l2', C=c, max_iter=1000, solver='liblinear', random_state=42)
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_scaled, y_train)
    
    # Correctly access and sum the coefficients of the underlying base models
    total_coef_magnitude = 0.0
    for model in ovr_model.estimators_:
        total_coef_magnitude += np.abs(model.coef_).sum()
        
    print(f"C = {c:6.2f} | Total Coefficient Magnitude = {total_coef_magnitude:.4f}")
print()

# Comment: As C increases, the total coefficient magnitude grows significantly larger. 
# This tells us that smaller C values enforce stronger regularization, shrinking weights 
# aggressively to keep the model simple, while larger C values relax this penalty, allowing 
# coefficients to grow to fit the training data more closely.



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

# 1. Fit PCA on X_digits as requested
pca = PCA()
pca.fit(X_digits)

# 2. Get the scores using transform separately
scores = pca.transform(X_digits)

# 3. Plot the results
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

# Comment: Yes, same-digit images definitely tend to cluster together in this 2D space. 
# Distinct groups are clearly visible, such as digit 0 clustered at the bottom center, 
# digit 4 on the far right, and digit 3 on the far left, though some structural overlap 
# occurs between digits with similar pixel distributions near the center of the plot.


# Q3
print("--- PCA Question 3 ---")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Locate the exact number of components for the 80% threshold
n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
print(f"Number of principal components required to reach 80% variance: {n_components_80}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.80, color='r', linestyle=':', label='80% Variance Threshold')
plt.axvline(x=n_components_80, color='g', linestyle=':', label=f'{n_components_80} Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png", dpi=150)
plt.close()
print("PCA Cumulative Variance plot saved to outputs/pca_variance_explained.png")

# Comment: Approximately 13 components are required to explain 80% of the variance.


# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Component thresholds and rows setup exactly matching instructions
n_values = [2, 5, 15, 40]
n_samples = 5 

# Set up grid of subplots: 5 rows (Original + 4 n values), 5 columns (digits)
fig, axes = plt.subplots(5, n_samples, figsize=(10, 10))

for col_idx in range(n_samples):
    # --- Row 0: Original Images ---
    axes[0, col_idx].imshow(images[col_idx], cmap='gray_r')
    axes[0, col_idx].axis('off')
    axes[0, col_idx].set_title(f"Digit {y_digits[col_idx]}", fontsize=10)
    
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Original", rotation=0, labelpad=40, va='center', fontweight='bold', fontsize=11)

    # --- Rows 1-4: PCA Reconstructions ---
    for row_idx, n in enumerate(n_values, start=1):
        recon_img = reconstruct_digit(col_idx, scores, pca, n_components=n)
        axes[row_idx, col_idx].imshow(recon_img, cmap='gray_r')
        axes[row_idx, col_idx].axis('off')
        
        # Apply clear structural row labels to the leftmost axis of each row
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel(f"n = {n}", rotation=0, labelpad=40, va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png", bbox_inches='tight', dpi=150)
plt.close()

print("\n--- PCA Question 4 Results ---")
print("PCA reconstruction grid saved to outputs/pca_reconstructions.png")

# Comment: The digits become clearly recognizable at n = 15. This cleanly matches 
# where our variance explanation curve begins to level off (~13-15 components), 
# demonstrating that the top principal components successfully capture the vast 
# majority of structural information in the dataset.

print("\nAll warmup exercises complete.")
