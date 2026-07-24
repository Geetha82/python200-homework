# Python 200: Advanced Predictive Modeling & Data Engineering

Welcome to my submission repository for Python 200. This repository houses foundational data analysis scripts, exploratory pipelines, and machine learning models developed throughout the course curriculum.

## Repository Architecture
```text
python200-homework/
├── assignments_01/           # Week 1 Data Engineering Fundamentals
│   ├── warmup_01.py          # Vectorization & basic syntax muscle memory
│   └── project_01.py         # Advanced Pandas processing & initial cleaning
├── assignments_02/           # Week 2 Predictive Linear Modeling
│   ├── warmup_02.py          # Scikit-Learn fit/predict mechanics & clustering
│   ├── project_02.py         # Multi-variable student math grade prediction
│   ├── student_performance_math.csv  # Semicolon-delimited source data
│   └── outputs/              # Diagnostic visual artifacts & plots
└── README.md                 # Primary directory guide
```

---

## 📈 Project 1: Data Engineering & Exploratory Foundations (Week 1)

### Core Objectives
*   Master structural vectorization principles using `NumPy` to eliminate nested iterative loops.
*   Manipulate raw structural matrices and parse complex multi-index `Pandas` DataFrame shapes.
*   Isolate target profiles via relational masks and prepare datasets for predictive modeling down the line.

### Key Implementation Details
*   **Vectorized Mechanics**: Implemented mathematical filters natively across feature arrays to ensure optimized runtime efficiency.
*   **Data Pipelines**: Built an algorithmic preprocessing sequence to handle missing records, align indices, and restructure messy categorical inputs into tabular profiles.

---

## 🤖 Project 2: Predictive Student Math Performance (Week 2)

### Core Objectives
*   Build an end-to-end multiple linear regression model utilizing the `scikit-learn` API (`create` → `fit` → `predict`).
*   Engineer numeric and binary variables from demographic, socio-environmental, and behavioral data.
*   Evaluate generalized model degradation under strict training and test holdout constraints (80/20 train/test split).

### Key Implementation Details
*   **Data Parsing Strategy**: Configured the raw data parsing engine to handle non-standard semicolon separators (`sep=";"`) and string token boundaries.
*   **Strategic Anomalous Cleaning**: Dropped rows containing `G3 == 0` because these represent quantitative situational exam absences rather than true academic tracking, eliminating massive non-linear distortion.
*   **Correlation-Driven Exploratory Data Analysis**: Computed global Pearson correlation coefficients across all background traits. Identified past class `failures` as the absolute strongest negative tracking metric ($\rho = -0.2938$) and mother's educational attainment (`Medu`) as the strongest positive indicator.
*   **Model Comparison & Evaluation**:
    *   **Baseline Model**: Built using `failures` alone, generating a test RMSE of ~3.1 grade points on a 0-20 scale.
    *   **Full Behavioral Model**: Incorporated all 11 core lifestyle variables (e.g., `studytime`, `higher`, `goout`). Increased explained variance ($R^2$) up to ~20.4%, dropping the typical prediction error (RMSE) down to 2.94 grade points.
*   **Data Leakage Analysis**: Experimented by injecting the first-period grade (`G1`) back into the features. This artificially pushed the test $R^2$ to ~0.80. Written internal script discussions prove this is an index proxy of capability rather than a causal driver, reinforcing why educators must utilize the lower-accuracy day-one behavioral model for proactive early-season interventions.

### Generated Diagnostic Visualizations
All artifacts are dynamically exported to the `assignments_02/outputs/` directory:
1.  `g3_distribution.png`: Histogram proving exam absence dropouts sit completely isolated from the standard Gaussian grading curve.
2.  `g3_vs_failures_boxplot.png`: Boxplot showing the stark downward median grade compression as historic failure limits increase.
3.  `g3_vs_medu_boxplot.png`: Boxplot validating the upward distribution shifts tied to higher maternal education.
4.  `predicted_vs_actual.png`: Visual alignment evaluation proving that the linear model safe-plays predictions toward the mean, underestimating the highest tiers and overestimating structural trailing edges.

---

## 🛠️ Execution & Environment Guide

Ensure you are operating inside your standard Python virtual environment, then execute your workflows directly from the respective subfolders to preserve internal path safety mappings:

### Running Week 1:
```bash
cd assignments_01
python warmup_01.py
python project_01.py
```

### Running Week 2:
```bash
cd assignments_02
python warmup_02.py
python project_02.py
```
