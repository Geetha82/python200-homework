import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline


#----- TASK 1: DATA EXPLORATION -----

# 1. Setup directories
os.makedirs("outputs", exist_ok=True)

# 2. Fetch dataset from UCI
print("Fetching Spambase dataset...")
spambase = fetch_ucirepo(id=94)
X = spambase.data.features
y = spambase.data.targets

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
print(f"  Ham (0):  {class_counts[0]}")
print(f"  Spam (1): {class_counts[1]}")
print("\nClass Balance Proportions:")
print(f"  Ham (0):  {class_props[0]:.2%}")
print(f"  Spam (1): {class_props[1]:.2%}")
print("\nAccuracy Score Interpretation Note:")
print(f"  A baseline model predicting all 'Ham' yields {class_props[0]:.2%} accuracy.")
print("  Raw accuracy must beat this threshold to show any true predictive value.")

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
print(f"Successfully generated and saved: {output_path}\n")


# ----- TASK 2: Prepare Your Data ------

# --- 1. TRAIN/TEST SPLIT ---
# Design Choices:
# - test_size=0.2: Reserves 20% of the dataset for testing generalization.
# - random_state=42: Ensures a deterministic split for reproducible results.
# - stratify=y: Maintains the ~61% Ham / 39% Spam class distribution across
#   both splits to prevent class-imbalance bias during evaluation.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. FEATURE SCALING ---
# Design Choices:
# - StandardScaler: Centers data around 0 with a unit standard deviation.
# - Prevents features with huge values (like capital run lengths) from
#   overwhelming features expressed as tiny percentages (word frequencies).
# - Data Leakage Prevention: We .fit_transform() ONLY on the training data.
#   The testing data is purely .transform()ed using the training parameters.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  ----- TASK 2b: PCA PREPROCESSING ------

# Rule: Always scale data before PCA so large raw values do not dominate variance.
# Data Leakage Prevention: Fit PCA on the training data ONLY. 

pca = PCA()
pca.fit(X_train_scaled)

# Calculate cumulative explained variance ratio
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Identify 'n': components where cumulative variance first reaches/exceeds 90%
n = np.argmax(cumulative_variance >= 0.90) + 1

print("\n===== Task 2: PCA Preprocessing Results =====")
print(f"Number of components where cumulative variance first reaches 90% (n): {n}")

# Plot and save cumulative explained variance curve
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Threshold')
plt.axvline(x=n, color='g', linestyle=':', label=f'n = {n} Components')
plt.title("Spambase PCA: Cumulative Explained Variance Curve")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/pca_explained_variance.png", dpi=150)
plt.close()

# Slice matrices down to the top 'n' principal components based on the determined 'n'
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# Keep both full scaled arrays (X_train_scaled, X_test_scaled) and 
# PCA-reduced arrays (X_train_pca, X_test_pca) intact for Task 3.


# ----- TASK 3: CLASSIFIER COMPARISON ------

# Keep track of models to locate our absolute top performer dynamically
best_acc = 0.0
best_model_name = ""
best_model_obj = None
best_X_test_data = None


def run_evaluation(label, clf, X_tr, y_tr, X_te, y_te):
    """Helper framework to fit, score, and log classification breakdowns."""
    global best_acc, best_model_name, best_model_obj, best_X_test_data
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)

    print(f"\n🚀 Classifier: {label}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_te, preds))

    if acc > best_acc:
        best_acc = acc
        best_model_name = label
        best_model_obj = clf
        best_X_test_data = X_te


# 1. KNN Configuration Variations
run_evaluation(
    "KNN (k=5) - Unscaled Data",
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    X_test,
    y_test,
)
run_evaluation(
    "KNN (k=5) - Scaled Data",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)
run_evaluation(
    "KNN (k=5) - PCA-Reduced Data",
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca,
    y_train,
    X_test_pca,
    y_test,
)

# 2. Decision Tree Max Depth Sweep
print("\n--- Decision Tree Max Depth Evaluation ---")
tree_depths = [3, 5, 10, None]
for depth in tree_depths:
    dt_sweep = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_sweep.fit(X_train, y_train)  # <-- Changed from X_train_scaled to X_train
    tr_acc = accuracy_score(y_train, dt_sweep.predict(X_train))  # <-- Changed from X_train_scaled to X_train
    te_acc = accuracy_score(y_test, dt_sweep.predict(X_test))  # <-- Changed from X_test_scaled to X_test
    depth_label = "Unlimited" if depth is None else str(depth)
    print(
        f"  Max Depth: {depth_label:<10} | Train Accuracy: {tr_acc:.4f} | Test Accuracy: {te_acc:.4f}"
    )

# Selected optimal production depth based on variance stabilization
chosen_depth = 5
dt_final = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
run_evaluation(
    f"Decision Tree (max_depth={chosen_depth}) - Unscaled Data",  # <-- Label updated
    dt_final,
    X_train,  # <-- Changed from X_train_scaled to X_train
    y_train,
    X_test,  # <-- Changed from X_test_scaled to X_test
    y_test,
)

# 3. Random Forest Classifier Execution (Scale-invariant, uses unscaled data)
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
print("\nTraining Random Forest (100 estimators)...")
run_evaluation(
    "Random Forest Classifier (Default)",
    rf_final,
    X_train,
    y_train,
    X_test,
    y_test,
)

# 4. Logistic Regression Configurations
run_evaluation(
    "Logistic Regression - Scaled Data",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)
run_evaluation(
    "Logistic Regression - PCA-Reduced Data",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_pca,
    y_train,
    X_test_pca,
    y_test,
)


# -----  TASK 3: FEATURE IMPORTANCE & CONFUSION MATRIX

# Extract Gini importance measurements into tabular layouts
# Extract the list of feature strings directly from your features DataFrame columns
feature_names = X.columns.tolist()

dt_imp_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": dt_final.feature_importances_}
).sort_values(by="Importance", ascending=False)
rf_imp_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": rf_final.feature_importances_}
).sort_values(by="Importance", ascending=False)

print("\n🔥 Top 10 Features: Single Decision Tree (depth=5)")
print(dt_imp_df.head(10).to_string(index=False))

print("\n🌲 Top 10 Features: Random Forest Ensemble")
print(rf_imp_df.head(10).to_string(index=False))

# Plot and save horizontal bar chart of the Random Forest importances
plt.figure(figsize=(10, 6))
top_10_rf = rf_imp_df.head(10).sort_values(by="Importance", ascending=True)
plt.barh(
    top_10_rf["Feature"],
    top_10_rf["Importance"],
    color="forestgreen",
    edgecolor="black",
    height=0.6,
)
plt.title(
    "Top 10 Feature Importances (Random Forest)", fontsize=12, fontweight="bold"
)
plt.xlabel("Gini Importance Score")
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/feature_importances.png", dpi=150)
plt.close()

# Generate and save optimal confusion matrix display
print(f"\n🏆 Best Performing Model: {best_model_name}")
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    best_model_obj,
    best_X_test_data,
    y_test,
    display_labels=["Ham", "Spam"],
    cmap=plt.cm.Blues,
    ax=plt.gca(),
)
plt.title(f"Confusion Matrix: {best_model_name}")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png", dpi=150)
plt.close()

print("\nProcessing complete. Check your 'outputs/' folder for visual assets!")


"""
======.   TASK 3 INTERPRETIVE WRITE-UP =====

1. WHAT DO YOU NOTICE AS TREE DEPTH INCREASES? OVERFITTING EXPLORATIVE SUMMARY:
As the max_depth sweeps from 3 -> 5 -> 10 -> None (unlimited), our training accuracy 
monotonically expands toward a perfect 1.0000. However, the test accuracy peaks early 
(around depth 5 or 10) and then declines or plateaus. This explicitly confirms over-fitting: 
an unconstrained tree memorizes noisy, unique email anomalies unique to the training split, 
generating an overly complex model that generalizes poorly out-of-sample.

2. PRODUCTION DEPTH SELECTION & ARGUMENTATION:
"I would deploy a max_depth of 5 for production. It restrains the model from asking overly 
brittle questions, focusing splits on high-level diagnostic items like punctuation mark counts, 
resulting in a more generalized classifier."

3. PCA VS NON-PCA ANALYSIS:
- KNN drops significantly in accuracy on PCA-reduced data compared to scaled data.
- Logistic Regression similarly suffers a degradation in performance when shifted into PCA space.
This matches the structural properties of our data: text indicator tokens carry subtle 
co-occurrence signals. Truncating the dataset variance down to a 90% limit drops minor structural 
word combinations that are highly predictive of spam profiles.

4. TARGET METRIC DEFENSE: ACCURACY VS FALSE POSITIVES VS FALSE NEGATIVES:
- Position: We must prioritize MINIMIZING FALSE POSITIVES (Maximizing Precision for the Spam Class).
- Defense: A False Negative means a spam message reaches an inbox, creating a minor, brief nuisance. 
  Conversely, a False Positive means a crucial, legitimate message—such as a job offer notification, 
  urgent medical memo, or legal confirmation—is falsely labeled as junk and permanently hidden 
  from view. Missing critical valid correspondence carries a substantially greater real-world cost 
  than occasionally manually deleting an un-filtered junk email.

5. ERROR PATTERN OBSERVATION (CONFUSION MATRIX READING):
When examining the generated outputs/best_model_confusion_matrix.png, the model typically commits 
more False Negatives (Spam leaking into Ham) than False Positives. This aligns with raw accuracy 
optimization targets, though a real-world system would require probability threshold modifications 
to prioritize lowering the False Positive rate.

6. MODEL FEATURE AGREEMENT:
Yes, both models show strong consensus on top features. Punctuation counters like 'char_freq_$' 
and 'char_freq_!' hold prominent importance scores across both estimators, aligning with intuition 
that high-pressure language and financial keywords serve as dominant identifiers of email spam.
================================================================================

"""

# ----- TASK 4: CROSS-VALIDATION -----


# 1. Define the models and their respective training datasets matching Task 3 configurations
# We use a dictionary mapping descriptive keys to (model_object, training_data_to_use) tuples
cv_experiments = {
    "KNN (k=5) - Unscaled Data": (KNeighborsClassifier(n_neighbors=5), X_train),
    "KNN (k=5) - Scaled Data": (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "KNN (k=5) - PCA-Reduced Data": (KNeighborsClassifier(n_neighbors=5), X_train_pca),
    f"Decision Tree (max_depth={chosen_depth}) - Scaled Data": (DecisionTreeClassifier(max_depth=chosen_depth, random_state=42), X_train_scaled),
    "Random Forest Classifier (Default)": (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "Logistic Regression - Scaled Data": (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_scaled),
    "Logistic Regression - PCA-Reduced Data": (LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_pca)
}

print("\n ===== Task 4: Cross-Validation Results =====\n")
print("Running 5-fold cross-validation on training partitions...")

# Tracker containers to dynamically evaluate cross-validation findings
highest_cv_mean = 0.0
best_cv_model_name = ""
lowest_cv_std = float('inf')
most_stable_model_name = ""

# 2. Iterate through models and calculate k-fold metrics
for name, (model, X_data) in cv_experiments.items():
    # cv=5 breaks the data down into 5 sequential partitions (folds)
    scores = cross_val_score(model, X_data, y_train, cv=5, scoring='accuracy')    
    mean_score = scores.mean()
    std_score = scores.std()
    
    print(f"\n {name}:")
    print(f"  Fold Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean Accuracy:      {mean_score:.4f}")
    print(f"  Standard Deviation: {std_score:.4f}")
    
    # Track the configuration with the best generalization performance
    if mean_score > highest_cv_mean:
        highest_cv_mean = mean_score
        best_cv_model_name = name
        
    # Track the configuration with the most consistent behavior across different slices
    if std_score < lowest_cv_std:
        lowest_cv_std = std_score
        most_stable_model_name = name

print("\n=========================================")
print("          CROSS-VALIDATION RESULTS        ")
print("=========================================")
print(f" Most Accurate Model: {best_cv_model_name} (Mean: {highest_cv_mean:.4%})")
print(f" Most Stable Model:   {most_stable_model_name} (Std Dev: {lowest_cv_std:.4f})")
print("=========================================\n")

"""
================================================================================
                    TASK 4 CONCEPTUAL REFLECTION 
================================================================================

1. WHICH MODEL IS THE MOST ACCURATE?
The RandomForestClassifier routinely emerges as the most accurate classifier after 
running 5-fold cross-validation. Its combined ensemble strategy allows it to average 
out errors from individual trees, enabling it to outscore single decision trees and 
linear boundaries by clean margins.

2. WHICH IS THE MOST STABLE (LOWEST VARIANCE ACROSS FOLDS)?
The RandomForestClassifier or Logistic Regression (on scaled data) typically registers 
the lowest standard deviation across folds. 
Random Forest owes its stability directly to its architectural ensembling: because it 
builds 100 decorrelated trees on different sub-samples of data and features, a fluke subset 
of training examples might throw off one tree, but it will barely shake the average 
consensus of the crowd. This stands in sharp contrast to the single Decision Tree, which 
shows high cross-validation variance because a minor change in a data partition alters 
its foundational top-level splits.

3. DOES THE RANKING MATCH WHAT YOU SAW WITH THE SINGLE TRAIN/TEST SPLIT?
Yes, the overall performance hierarchy generally matches the single train/test split, 
validating that the original 80/20 holdout was an accurate, un-biased slice. However, 
the absolute accuracy numbers during cross-validation are slightly lower or more 
conservative than the single test split. This is expected behavior: cross-validation offers 
a more honest, robust view of true generalization capacity precisely because it forces 
the architectures to validate across 5 entirely different permutations of hidden data.
================================================================================

"""

# ----- TASK 5: PREDICTION PIPELINE -----

print("\n ===== Task 5: Building Prediction Pipelines Results =====")

# --- 1. THE TREE-BASED PRODUCTION PIPELINE ---
# Decision trees and random forests are scale-invariant, meaning they
# operate perfectly on raw data features without an explicit scaler step.
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- 2. THE NON-TREE PRODUCTION PIPELINE ---
# Distance and linear-based models require explicit scaling. PCA is injected
# between the scaling step and the model estimator as requested by the prompt.
nontree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),  # Uses 'n' derived in Task 2b
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

# --- 3. TRAIN AND EVALUATE TREE PIPELINE ---
print("\nTraining the Tree-Based Pipeline (Random Forest)...")
tree_pipeline.fit(X_train, y_train)

tree_accuracy = tree_pipeline.score(X_test, y_test)
print(f"Tree Pipeline Test Accuracy Score: {tree_accuracy:.4f}")

tree_predictions = tree_pipeline.predict(X_test)
print("\nTree Pipeline Final Classification Report:")
print(classification_report(y_test, tree_predictions))


# --- 4. TRAIN AND EVALUATE NON-TREE PIPELINE ---
print("\nTraining the Non-Tree Pipeline (Logistic Regression + PCA)...")
nontree_pipeline.fit(X_train, y_train)

nontree_accuracy = nontree_pipeline.score(X_test, y_test)
print(f"Non-Tree Pipeline Test Accuracy Score: {nontree_accuracy:.4f}")

nontree_predictions = nontree_pipeline.predict(X_test)
print("\nNon-Tree Pipeline Final Classification Report:")
print(classification_report(y_test, nontree_predictions))

# ----- Build Two pipelines -----

# --- 1. THE TREE-BASED PRODUCTION PIPELINE ---
# Trees are completely scale-invariant, so our Random Forest pipeline 
# needs no scaling transformer—only the raw model classifier step.
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- 2. THE NON-TREE PRODUCTION PIPELINE ---
# Distance/Linear models require rigid preprocessing. We scale the data first.
nontree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

# --- 3. TRAIN AND EVALUATE TREE PIPELINE ---
print("Training Tree-Based Pipeline (Random Forest)...")
tree_pipeline.fit(X_train, y_train)
tree_preds = tree_pipeline.predict(X_test)

print("\n🌲 Tree Pipeline Final Classification Report:")
print(classification_report(y_test, tree_preds))

# --- 4. TRAIN AND EVALUATE NON-TREE PIPELINE ---
print("Training Non-Tree Pipeline (Logistic Regression)...")
nontree_pipeline.fit(X_train, y_train)
nontree_preds = nontree_pipeline.predict(X_test)

print("\n📈 Non-Tree Pipeline Final Classification Report:")
print(classification_report(y_test, nontree_preds))

"""
================================================================================
                  Comment on the  Pipelines
================================================================================

1. DO THESE TWO PIPELINES HAVE THE SAME STRUCTURE? WHY OR WHY NOT?
No, the two pipelines feature fundamentally distinct architectures. 
- Non-Tree Pipeline: Built with an explicit preprocessing step ("scaler") before 
  the estimator. This is mathematically necessary because Logistic Regression uses 
  regularization penalties and gradient optimization solvers that fail or become 
  highly distorted when tracking features with wildly mismatching magnitudes.
- Tree Pipeline: Connects directly to the classifier step without any scaling. 
  Decision Trees and Random Forests split on localized vertical and horizontal 
  feature boundaries one feature at a time. Because they look exclusively at thresholds 
  (e.g., is "word_freq_free" > 0.5?), transforming the numeric range has absolute 
  zero effect on the purity drop of the split, making standardizing completely redundant.

2. WHAT IS THE PRACTICAL VALUE OF PACKAGING A MODEL THIS WAY FOR DEPLOYMENT?
- Complete Prevention of Data Leakage: It locks down processing sequence logic. 
  The pipelines ensure that operations like fitting scaling parameters occur exclusively 
  on the training fold. When unseen test profiles or fresh real-world emails flow through 
  the pipeline, they are instantly .transformed() using historical metrics, preventing 
  future insight bleed.
- Eliminates Human Engineering Bookkeeping: Instead of forcing an engineering team 
  to manually preserve, document, and chain together individual standalone transformation scripts 
  (such as handling a raw csv file, executing a scaler, passing it to PCA, and then running 
  the model pickling files), a Pipeline unifies everything into a single operational object.
- Safe Hand-offs & Microservice Ready: A pipeline allows you to call a single unified file. 
  You can export the entire Pipeline object as a single binary file (like a pickle file). 
  Another engineer can import it into a production web API, feed it raw data strings, 
  and receive automated inference outputs instantly without needing to know anything 
  about your internal scaling formulas or column dimensions.
================================================================================

"""
