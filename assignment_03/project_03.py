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


# ----- TASK 2: DATA PREPARATION ------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --- 1. TRAIN/TEST SPLIT CONSTRAINTS ---
# Design Choices:
# - test_size=0.2: Reserves 20% of the dataset for testing generalization.
# - random_state=42: Ensures a deterministic split for reproducible results.
# - stratify=y: Maintains the ~61% Ham / 39% Spam class distribution across
#   both splits to prevent class-imbalance bias during evaluation.

print("\n ===== TASK 2: DATA PREPARATION RESULTS =====\n")

print("Splitting data into training and validation sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.2, random_state=42, stratify=y
)

print(f"  Training Features Shape: {X_train.shape}")
print(f"  Testing Features Shape:  {X_test.shape}")


# --- 2. FEATURE SCALING (STANDARDIZATION) ---
# Design Choices:
# - StandardScaler: Centers data around 0 with a unit standard deviation.
# - Prevents features with huge values (like capital run lengths) from
#   overwhelming features expressed as tiny percentages (word frequencies).
# - Data Leakage Prevention: We .fit_transform() ONLY on the training data.
#   The testing data is purely .transform()ed using the training parameters.

print("\nScaling features using StandardScaler to neutralize scale disparities...")
scaler = StandardScaler()

# Fit to training features and transform them
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing features using the already-fit scaler parameters
X_test_scaled = scaler.transform(X_test)

print("  Feature scaling complete.")
print("  Pre-scaled min/max for capital_run_length_total:", int(X.iloc[:, -1].min()), "/", int(X.iloc[:, -1].max()))
print(f"  Post-scaled mean/std for feature 0: {X_train_scaled[:, 0].mean():.2f} / {X_train_scaled[:, 0].std():.2f}")


# ----- PCA PREPROCESSING -----


# --- 1. FIT PCA ON SCALED TRAINING DATA ONLY ---
# Fitting on X_train_scaled only prevents test-set information leakage.
# Leaving n_components empty calculates all possible principal components first.
pca = PCA()
pca.fit(X_train_scaled)

# --- 2. CALCULATE CUMULATIVE EXPLAINED VARIANCE ---
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find 'n': the number of components where cumulative variance first reaches or exceeds 90%
# We add 1 because Python index positions start at 0
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("\n ===== PCA preprocessing Results =====\n")

print(f"Total original features: {X_train_scaled.shape[1]}")
print(f"Number of PCA components needed to reach 90% variance (n): {n_components_90}")
print(f"Exact cumulative variance captured at n={n_components_90}: {cumulative_variance[n_components_90 - 1]:.4f}")

# --- 3. PLOT AND SAVE THE VARIANCE CURVE ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b', label='Cumulative Variance')
plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Threshold')
plt.axvline(x=n_components_90, color='g', linestyle=':', label=f'n = {n_components_90}')

plt.title("Spambase PCA: Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)

# Save visualization artifact
plt.tight_layout()
output_pca_path = "outputs/pca_explained_variance.png"
plt.savefig(output_pca_path, dpi=150)
plt.close()
print(f"Saved PCA variance curve plot to: {output_pca_path}")

# --- 4. TRANSFORM AND SLICE BOTH DATASETS ---
# Slicing down to index [: , :n_components_90] isolates our top 'n' dimensions
X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]

print(f"\nPCA Feature Reductions Complete:")
print(f"  X_train_pca Shape: {X_train_pca.shape}")
print(f"  X_test_pca Shape:  {X_test_pca.shape}")

# ----- TASK 3: CLASSIFIER COMPARISON -----

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Trackers to dynamically find and map the best model configuration for the confusion matrix
best_accuracy = 0.0
best_model_name = ""
best_model_obj = None
best_X_test = None

# Extract the clean feature names from the original dataframe columns
feature_names = X.columns.tolist()


def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Utility helper to fit, evaluate, print classification reports,

    and track top performers.
    """
    global best_accuracy, best_model_name, best_model_obj, best_X_test

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)

    print(f"\n Classifier: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_te, preds))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model_obj = model
        best_X_test = X_te


# --- 1. KNN ON UNSCALED DATA ---
evaluate_model(
    "KNN (k=5) - Unscaled Data",
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    X_test,
    y_test,
)

# --- 2. KNN ON SCALED VS PCA-REDUCED DATA ---
evaluate_model(
    "KNN (k=5) - Scaled Data",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

evaluate_model(
    "KNN (k=5) - PCA-Reduced Data",
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca,
    y_train,
    X_test_pca,
    y_test,
)

# --- 3. DECISION TREE DEPTH TUNING ---
print("\n--- Decision Tree Max Depth Tuning ---")
depths = [3, 5, 10, None]
for depth in depths:
    dt_tune = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_tune.fit(X_train_scaled, y_train)
    tr_acc = accuracy_score(y_train, dt_tune.predict(X_train_scaled))
    te_acc = accuracy_score(y_test, dt_tune.predict(X_test_scaled))
    depth_str = "Unlimited" if depth is None else str(depth)
    print(
        f"  Max Depth: {depth_str:<10} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}"
    )

# Establish chosen depth for production based on variance tuning performance
chosen_depth = 5
dt_final = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
evaluate_model(
    f"Decision Tree (max_depth={chosen_depth}) - Scaled Data",
    dt_final,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

# --- 4. RANDOM FOREST CLASSIFIER ---
# Trained on unscaled data since trees are scale invariant
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
print("\nTraining Random Forest Classifier (100 trees)...")
evaluate_model(
    "Random Forest Classifier (Default)",
    rf_final,
    X_train,
    y_train,
    X_test,
    y_test,
)

# --- 5. LOGISTIC REGRESSION ON SCALED VS PCA-REDUCED DATA ---
evaluate_model(
    "Logistic Regression - Scaled Data",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

evaluate_model(
    "Logistic Regression - PCA-Reduced Data",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_pca,
    y_train,
    X_test_pca,
    y_test,
)

print("\n ===== Task 3: A Classifier Comparison Results =====\n")

# --- 6. ARTIFACT GENERATION: CONFUSION MATRIX ---
print(
    f"\n Best Performing Configuration Identified: {best_model_name} ({best_accuracy:.4%})"
)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    best_model_obj,
    best_X_test,
    y_test,
    display_labels=["Ham", "Spam"],
    cmap=plt.cm.Blues,
    ax=plt.gca(),
)
plt.title(f"Confusion Matrix:\n{best_model_name}")
plt.tight_layout()
output_cm_path = "outputs/best_model_confusion_matrix.png"
plt.savefig(output_cm_path, dpi=150)
plt.close()
print(f"Saved optimal configuration confusion matrix to: {output_cm_path}")

print("\n ===== FEATURE IMPORTANCE ANALYSIS =====\n")

# Extract importance vectors into dataframes
dt_importances = pd.DataFrame(
    {"Feature": feature_names, "Importance": dt_final.feature_importances_}
).sort_values(by="Importance", ascending=False)

rf_importances = pd.DataFrame(
    {"Feature": feature_names, "Importance": rf_final.feature_importances_}
).sort_values(by="Importance", ascending=False)

# Print Top 10 lists for comparative review
print("\n Top 10 Features: Decision Tree (max_depth=5)")
print(dt_importances.head(10).to_string(index=False))

print("\n Top 10 Features: Random Forest (100 Trees)")
print(rf_importances.head(10).to_string(index=False))

# Plot horizontal bar chart of the Random Forest importances
plt.figure(figsize=(10, 6))
top_rf = rf_importances.head(10).sort_values(by="Importance", ascending=True)

plt.barh(
    top_rf["Feature"],
    top_rf["Importance"],
    color="forestgreen",
    edgecolor="black",
    height=0.6,
)
plt.title(
    "Top 10 Most Important Features (Random Forest Classifier)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Gini Importance Score (Mean Decrease in Impurity)")
plt.ylabel("Dataset Feature Name")
plt.grid(axis="x", linestyle=":", alpha=0.6)

# Save chart artifact
plt.tight_layout()
output_fi_path = "outputs/feature_importances.png"
plt.savefig(output_fi_path, dpi=150)
plt.close()
print(
    f"\nSaved Random Forest feature importance bar chart to: {output_fi_path}"
)

"""
================================================================================
                FEATURE IMPORTANCE REFLECTION & INTUITION
================================================================================

1. DO THE TWO MODELS AGREE ON WHICH FEATURES MATTER MOST?
- High-Level Agreement: Yes, both models heavily prioritize specific punctuation marks 
  and characters, such as 'char_freq_$' (dollar signs) and 'char_freq_!' (exclamation marks).
- Granular Disagreement: The single Decision Tree concentrates almost all its structural 
  importance onto the top 2 or 3 features because it relies on early, aggressive splits. 
  Conversely, the Random Forest spreads its importance scores more smoothly across a broader 
  mix of word indicators (like 'word_freq_remove', 'word_freq_free', and capital sequences). 
  This happens because Random Forest forces trees to split on random feature subsets, allowing 
  secondary features to prove their predictive value.

2. DO THE RESULTS MATCH YOUR INTUITION ABOUT SPAM?
Yes, the results align perfectly with real-world intuition:
- 'char_freq_$' links directly to financial phishing and scam offers.
- 'char_freq_!' indicates aggressive marketing copy and high-pressure text.
- 'word_freq_remove' points to automated unsubscribe text commonly found in bulk mail list footers.
- 'capital_run_length_longest' represents shouting behaviors (ALL CAPS lines) typical of junk mail.

3. MODEL STABILITY PREDICTIONS FOR TASK 4 CROSS-VALIDATION:
Because the Decision Tree relies heavily on a few specific features, slight changes in its training 
slice will drastically alter its structure, resulting in high validation variance. The Random Forest 
averages out the decisions of 100 decorrelated trees, making it structurally stable and lowering 
its variance across different evaluation data folds.
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

# --- 1. DEFINE THE UNIFIED PIPELINE CONTEXT ---
# Design Choice:
# By enclosing the components inside a scikit-learn Pipeline object, data is 
# automatically passed in order through scaling, then PCA reduction, and finally 
# into the LogisticRegression model estimator.
#
# Crucially, this structurally eliminates the risk of Data Leakage because the 
# pipeline isolates fitting mechanics to training inputs only.
production_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components_90)), # Uses 'n' derived in Task 2b
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

# --- 2. TRAIN THE PIPELINE ---
# We pass the raw unscaled X_train here. The pipeline handles all 
# bookkeeping internally, fitting the scaler and PCA elements safely.

print("\n ===== Task 5: Building a Prediction Pipeline Results =====\n")

print("Training the unified production pipeline...")
production_pipeline.fit(X_train, y_train)

# --- 3. EVALUATING VIA PIPELINE ACTIONS ---
# Use the built-in score() method to compute raw test accuracy in one single call.
# This runs X_test through the scaling/PCA transformation formulas learned from 
# X_train, runs predictions, and compares them to y_test.
pipeline_accuracy = production_pipeline.score(X_test, y_test)
print(f"Pipeline Test Accuracy Score: {pipeline_accuracy:.4f}")

# Generate the full classification report to verify pipeline integrity
pipeline_predictions = production_pipeline.predict(X_test)
print("\nPipeline Final Classification Report:")
print(classification_report(y_test, pipeline_predictions))

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
