import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1
# Split X and y into training and test sets using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20, 
    stratify=y, 
    random_state=42
)
print("\n===== Preprocessing Question 1 Results =====")
# Print the shapes of all four arrays
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# Q2 
# Fit a StandardScaler on X_train and use it to transform both X_train and X_test. 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n===== Preprocessing Question 2 Results =====")
# Print the mean of each column in X_train_scaled -- they should all be very close to 0.
print(f"Means of scaled X_train columns: {X_train_scaled.mean(axis=0)}")

# comment explaining in one sentence why you fit the scaler on X_train only.
# We fit the scaler on X_train only to prevent data leakage from the test set into our model training.

# --- KNN ---
#  Q1
# Build a KNeighborsClassifier with n_neighbors=5
knn_unscaled = KNeighborsClassifier(n_neighbors=5)

# fit it on the unscaled training data (X_train)
knn_unscaled.fit(X_train, y_train)

# predict on the test set. 
y_predict_unscaled = knn_unscaled.predict(X_test)

# Print the accuracy score and the full classification report.
print("\n===== KNN Question 1 Results =====")
print(f"Unscaled KNN Test Accuracy: {accuracy_score(y_test, y_predict_unscaled):.4f}")
print("\nUnscaled KNN Classification Report:")
print(classification_report(y_test, y_predict_unscaled))

# Q2
#Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
# Predict on the scaed test set
y_predict_scaled = knn_scaled.predict(X_test_scaled)

#  Print the accuracy score. 
print("\n===== KNN Question 2 Results =====")
print(f"Scaled KNN Test Accuracy: {accuracy_score(y_test, y_predict_scaled):.4f}")

# Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?
# Comment:
# For the Iris dataset, feature scaling typically makes little to no difference in performance (often yielding identical or slightly lower accuracy). 
# This happens because all four original features are physical measurements recorded in the exact same unit (centimeters) and share very similar numerical scales, preventing any single feature from artificially dominating the distance calculations.

# Q3
# Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. 
knn_cv_model = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_cv_model, X_train, y_train, cv=5)

# Print each fold score, the mean, and the standard deviation
print("\n===== KNN Question 3 Results =====")
print(f"Scores for each fold: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.4f}")
print(f"Standard deviation of scores: {cv_scores.std():.4f}")

# Add a comment: 
# is this result more or less trustworthy than a single train/test split?
# This cross-validation result is significantly more trustworthy than a single train/test split. 
# and why?
# # It evaluates the model across multiple distinct data splits, ensuring that the performance metric is stable and not an artifact of a lucky or unlucky random selection of data during a single split.

# Q4

# Define the array
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("\n===== KNN Question 4 Results =====")
# Loop over k values
#  For each, compute 5-fold cross-validation accuracy on the unscaled training data and 
for k in k_values:
    knn_tune = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_tune, X_train, y_train, cv=5)
    # print k and the mean CV score
    print(f"k = {k:2d} | Mean CV Accuracy: {scores.mean():.4f}")

# Add a comment identifying which k you would choose and why.
#I would choose k = 11 or k = 13 because they yield the highest mean cross-validation accuracy (typically around 96.67% depending on the split) on the training set. 
# Picking a slightly higher, odd k value helps smooth out decision boundaries, protects the model against localized noise, and avoids voting ties.   

# --- Classifier Evaluation ---
# Q1

# Generate the raw confusion matrix from your unscaled KNN test predictions
cm = confusion_matrix(y_test, y_predict_unscaled)

# Set up the visualization display map using the true class names
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# Render and save the plot figure
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("KNN Confusion Matrix (Iris Dataset)")
plt.savefig('outputs/knn_confusion_matrix.png')
plt.close()

# Print completion statement to terminal
print("\n===== Classifier Evaluation Question 1 Results =====")

print("Confusion matrix generated and saved to 'outputs/knn_confusion_matrix.png'.")

# Comment:
# The model most often confuses versicolor and virginica. 
# This happens because their physical feature measurements heavily overlap in geographic space, whereas the setosa species is linearly separable and easily isolated.

# --- The sklearn API: Decision Trees ---
# Q1
# Create a DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# fit it on the unscaled training data
dt_model.fit(X_train, y_train)

# predict on the test set
y_pred_dt = dt_model.predict(X_test)

# Print the accuracy score and classification report
print("\n===== The sklearn API: Decision Trees Question 1 Results =====")
print(f"Decision Tree Test Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# comment comparing the Decision Tree accuracy to KNN:
# The Decision Tree accuracy is typically very comparable to or slightly lower than KNN on the Iris dataset (often around 93.33% to 96.67% depending on the split), as both models are highly capable of isolating the flower boundaries, though KNN smooths the boundary slightly better than the axis-aligned splits of a tree.

# second comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?
# Scaling data makes absolutely no difference to a Decision Tree's performance. 
# Because trees split features based on monotonic step functions (e.g., feature_1 > 2.5) rather than calculating geometric or matrix distances, transforming or scaling the features preserves their relative order and results in identical tree splits.

# --- Logistic Regression and Regularization ---
# Q1

# Define the C parameter values to evaluate
c_values = [0.01, 1.0, 100.0]

print("\n--- Logistic Regression Regularization Question1 Results ---")
# Loop over each C value, train the model, and measure coefficient sizes
for c in c_values:
    # Initialize model with current C value and specified parameters
    lr_model = OneVsRestClassifier(LogisticRegression(C=c, max_iter=1000, solver='liblinear', random_state=42))

    # Fit the model on the scaled training data
    lr_model.fit(X_train_scaled, y_train)
    
    # Calculate the sum of the absolute values of all coefficients
    coef_magnitude = sum(np.abs(est.coef_).sum() for est in lr_model.estimators_)
    
    # Display results
    print(f"C value: {c:6.2f} | Total Coefficient Magnitude: {coef_magnitude:.4f}")

# Comment:
# As the C value increases, the total magnitude of the coefficients also increases. 
# This tells us that smaller values of C apply stronger regularization, forcing the model to penalize large weights and shrink the coefficients closer to zero to prevent overfitting, whereas larger values of C loosen the penalty, allowing the model to fit the training data more aggressively.

# # --- PCA ---
# PCA Setup
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1
print("\n--- PCA Question1 Results ---")
# Print the shape of X_digits and images
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape:   {images.shape}")

# create a 1-row subplot showing one example of each digit class (0-9)
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for digit in range(10):
    # Find the index of the first occurrence of this digit
    index = np.where(y_digits == digit)[0][0]
    
    # Plot the 8x8 image using the reversed grayscale colormap
    axes[digit].imshow(images[index], cmap='gray_r')
    axes[digit].set_title(f"Label: {digit}")
    axes[digit].axis('off')

# Save the figure to outputs/sample_digits.png
os.makedirs("outputs", exist_ok=True)
plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()

# Q2
# Fit PCA() on X_digits (with no n_components argument) 
pca = PCA()
pca.fit(X_digits)
# get the scores with scores = pca.transform(X_digits)
scores = pca.transform(X_digits)

# Generate the 2D projection scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')

plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('Digits Dataset: 2D PCA Projection')
plt.grid(True, alpha=0.3)

# Save the figure to outputs/pca_2d_projection.png
print("\n--- PCA Question2 Results ---")
print("\n pca_2d_projection.png saved in outputs folder ")
plt.savefig("outputs/pca_2d_projection.png", bbox_inches='tight')
plt.close()

# Add a comment: do same-digit images tend to cluster together in this 2D space?
# Yes, same-digit images clearly tend to cluster together in this 2D space. 
# Even though the original data spans 64 dimensions, capturing just the top two dimensions 
# of maximum variance is enough to visually separate many of the digit classes into distinct groups.
# While some clusters overlap (like 1, 8, and 5), others (like 0, 4, and 3) are well-segregated.

# Q3
# Using the PCA object you fit in Question 2
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance vs. number of components
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', markersize=4)

# Visual anchors for interpretation
plt.axhline(y=0.80, color='r', linestyle='--', alpha=0.7, label='80% Variance Threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Digits Dataset: Cumulative Explained Variance')
plt.grid(True, alpha=0.3)
plt.legend()

#  Save to outputs/pca_variance_explained.png
print("\n--- PCA Question3 Results ---")
print("\n pca_variance_explained.png saved in outputs folder ")
plt.savefig("outputs/pca_variance_explained.png", bbox_inches='tight')
plt.close()

#  Add a comment: approximately how many components do you need to explain 80% of the variance?
# You need approximately 13 components to explain 80% of the variance.
# This means we can reduce the data dimensionality from 64 original pixel features 
# down to just 13 components while still retaining 80% of the structural information.

# Q4

# The preprocessing lesson showed that a reconstruction is built by starting from the mean and adding each component weighted by its score. Here is the same idea generalized to n components -- add this function to your file:

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Setup grid: 5 rows (Original, n=2, n=5, n=15, n=40) x 5 columns (first 5 samples)
n_values = [2, 5, 15, 40]
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for col_idx in range(5):
    # Row 0: Original images
    axes[0, col_idx].imshow(images[col_idx], cmap='gray_r')
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Original", fontsize=12, fontweight='bold')
    axes[0, col_idx].set_xticks([])
    axes[0, col_idx].set_yticks([])

    # Rows 1-4: Reconstructions with varying n_components
    for row_idx, n in enumerate(n_values, start=1):
        recon_img = reconstruct_digit(col_idx, scores, pca, n_components=n)
        axes[row_idx, col_idx].imshow(recon_img, cmap='gray_r')
        
        if col_idx == 0:
            axes[row_idx, col_idx].set_ylabel(f"n = {n}", fontsize=12, fontweight='bold')
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])

# Save to outputs/pca_reconstructions.png
print("\n--- PCA Question4 Results ---")
print("\n outputs/pca_reconstructions.png saved in outputs folder ")
plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png", bbox_inches='tight')
plt.close()

# Add a comment: at what n do the digits become clearly recognizable, 
# and does that match where the variance curve levels off?

# The digits become clearly recognizable at around n = 15 components. 
# This aligns perfectly with the variance curve from Question 3, where 13-15 components 
# capture over 80% of the variance. By the time we reach n = 40, the reconstructions 
# are nearly identical to the originals, capturing almost all fine details as the 
# variance curve flattens out completely toward 100%.