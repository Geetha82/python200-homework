# 🌍 World Happiness Pipeline (2015–2024)

## Project Overview
This project is an end-to-end **Data Engineering and Analysis pipeline** built using **Prefect**. It automates the extraction of a decade of global happiness data, performs data cleaning, conducts statistical hypothesis testing, and generates visual insights.

The analysis specifically focuses on global well-being trends, the impact of the **COVID-19 pandemic** (Pre-2020 vs. Post-2020), and the socioeconomic drivers of happiness.

## 🛠️ Technology Stack
*   **Orchestration**: [Prefect](https://prefect.io)
*   **Data Processing**: Pandas, NumPy
*   **Statistical Testing**: SciPy (T-tests, Pearson Correlation)
*   **Visualization**: Matplotlib, Seaborn

## 📊 The "Six Key Indicators"
The pipeline analyzes six core variables that determine a nation's **Happiness Score (Ladder Score)**:
1.  **GDP per capita**: Economic strength.
2.  **Social support**: Having someone to count on in times of trouble.
3.  **Healthy life expectancy**: Physical well-being.
4.  **Freedom**: The freedom to make life choices.
5.  **Generosity**: Based on recent charitable donations.
6.  **Perceptions of corruption**: Trust in government and business.

## 🚀 Getting Started

### 1. Installation
Install the necessary Python libraries:
```bash
pip install prefect pandas matplotlib seaborn scipy
```

### 2. Launch the Dashboard
Start the local Prefect server to monitor the pipeline in your browser:
```bash
prefect server start
```
*URL: http://127.0.0.1:4200*

### 3. Run the Pipeline
Execute the script from your project root:
```bash
python3 assignments_01/project_01.py
```

## 📁 Pipeline Architecture
The script executes six orchestrated tasks:
*   **Task 1 (ETL)**: Downloads 10 yearly CSVs from GitHub, adds a `year` column, and creates `merged_happiness.csv`.
*   **Task 2 (Stats)**: Computes global mean, median, and regional happiness rankings.
*   **Task 3 (Visuals)**: Generates distribution histograms, yearly boxplots, and correlation heatmaps.
*   **Task 4 (Hypothesis)**: Performs an independent t-test comparing **2019 vs. 2020** happiness.
*   **Task 5 (Correction)**: Runs correlations using the **Bonferroni correction** to prevent false positives.
*   **Task 6 (Report)**: Logs a plain-language summary of findings to the Prefect Dashboard.

## 📂 Output Files (`assignments_01/outputs/`)
*   `merged_happiness.csv`: The complete 10-year master dataset.
*   `yearly_statistics.csv`: Mean and standard deviation by year.
*   `happiness_trend.png`: Line plot showing happiness over time.
*   `correlation_heatmap.png`: Matrix of relationships between all indicators.
*   `gdp_vs_happiness.png`: Scatter plot of wealth vs. well-being.

## 📝 Key Findings
*   **Predictive Power**: Social Support and GDP consistently show the strongest positive correlations with happiness.
*   **Resilience**: Global happiness scores remained remarkably stable during the 2020 pandemic onset compared to 2019.
*   **Statistical Rigor**: By applying the Bonferroni correction, the pipeline confirms which socioeconomic factors are truly significant.
