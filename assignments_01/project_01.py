import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import flow, task, get_run_logger

# ===== Task 1: Load Multiple Years of Data =====
@task(retries=3, retry_delay_seconds=2)
def load_all_happiness_data(base_url, years):
    logger = get_run_logger()
    all_data = []
    
    # SETUP PATHS: Ensures the CSV lands in assignments_01/outputs/
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        logger.info(f"Loading Year: {year}")
        url = f"{base_url}{year}.csv"
        
        # Quirks: Semicolon separator and comma decimals
        df = pd.read_csv(url, sep=";", decimal=",")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        
        # Add year column (Requirement)
        df["year"] = int(year)
        all_data.append(df)

    # Merge and Save
    merged_df = pd.concat(all_data, ignore_index=True)
    output_file = os.path.join(output_dir, "merged_happiness.csv")
    merged_df.to_csv(output_file, index=False)
    
    logger.info(f"Task 1 Complete: Merged data saved to {output_file}")
    return merged_df

# ===== Task 2: Descriptive Statistics =====
@task
def descriptive_statistics(df: pd.DataFrame):
    logger = get_run_logger()

    # ----------------------------
    # Overall statistics
    # ----------------------------
    mean_score = df["happiness_score"].mean()
    median_score = df["happiness_score"].median()
    std_score = df["happiness_score"].std()

    logger.info("=== Overall Happiness Score Stats ===")
    logger.info(f"Mean: {mean_score}")
    logger.info(f"Median: {median_score}")
    logger.info(f"Std Dev: {std_score}")

    # ----------------------------
    # Group by year
    # ----------------------------
    year_mean = df.groupby("year")["happiness_score"].mean()

    logger.info("=== Mean Happiness by Year ===")
    logger.info(f"\n{year_mean}")

    # ----------------------------
    # Group by region
    # ----------------------------
    region_mean = df.groupby("regional_indicator")["happiness_score"].mean().sort_values(ascending=False)

    logger.info("=== Mean Happiness by Region ===")
    logger.info(f"\n{region_mean}")

    # Return everything for later tasks
    return {
        "overall_mean": mean_score,
        "overall_median": median_score,
        "overall_std": std_score,
        "year_mean": year_mean,
        "region_mean": region_mean
    }

# ===== Task 3: Visual Exploration =====
@task
def create_visualizations(df):
    logger = get_run_logger()

    # Output directory (inside assignments_01)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # 1. Histogram
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["happiness_score"], bins=20)
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")

    histogram_path = os.path.join(output_dir, "happiness_histogram.png")
    plt.savefig(histogram_path)
    plt.close()

    logger.info(f"Saved histogram to {histogram_path}")
     
    # ----------------------------
    # 2. Boxplot by Year
    # ----------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="year", y="happiness_score")

    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")

    boxplot_path = os.path.join(output_dir, "happiness_by_year.png")
    plt.savefig(boxplot_path)
    plt.close()

    logger.info(f"Saved boxplot to {boxplot_path}")

    # ----------------------------
    # 3. GDP vs Happiness Scatter Plot
    # ----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(df["gdp_per_capita"], df["happiness_score"])

    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")

    scatter_path = os.path.join(output_dir, "gdp_vs_happiness.png")
    plt.savefig(scatter_path)
    plt.close()

    logger.info(f"Saved scatter plot to {scatter_path}")

    # ----------------------------
    # 4. Correlation Heatmap
    # ----------------------------
    plt.figure(figsize=(10, 8))

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=True)

    plt.title("Correlation Heatmap")

    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    logger.info(f"Saved heatmap to {heatmap_path}")

    
# ===== Task 4: Hypothesis Testing  =====
@task
def run_hypothesis_tests(df):
    logger = get_run_logger()
    target = "happiness_score"

    # --- TEST 1: 2019 vs 2020 (Pandemic Impact) ---
    data_2019 = df[df['year'] == 2019][target]
    data_2020 = df[df['year'] == 2020][target]
    
    # Independent t-test
    t_stat, p_val = stats.ttest_ind(data_2019, data_2020, nan_policy='omit')
    
    mean_19 = data_2019.mean()
    mean_20 = data_2020.mean()

    logger.info("--- T-Test: Global Happiness 2019 vs 2020 ---")
    logger.info(f"2019 Mean: {mean_19:.4f} | 2020 Mean: {mean_20:.4f}")
    logger.info(f"t-statistic: {t_stat:.4f} | p-value: {p_val:.4f}")

    # Plain-language interpretation
    if p_val < 0.05:
        interp = (f"With a p-value of {p_val:.4f}, the difference is statistically significant. "
                  "This suggests that global happiness scores shifted measurably during the "
                  "first year of the pandemic.")
    else:
        interp = (f"With a p-value of {p_val:.4f}, the difference is not statistically significant. "
                  "Despite the global crisis, average happiness scores remained remarkably "
                  "resilient and stable between 2019 and 2020.")
    logger.info(f"Interpretation: {interp}")

    # --- TEST 2: Regional Comparison (North America vs. Sub-Saharan Africa) ---
    # Comparing two regions we expect to differ based on Task 2 stats
    reg_high = df[df['regional_indicator'] == 'North America and ANZ'][target]
    reg_low = df[df['regional_indicator'] == 'Sub-Saharan Africa'][target]
    
    t_reg, p_reg = stats.ttest_ind(reg_high, reg_low, nan_policy='omit')
    
    logger.info("--- T-Test: Regional Comparison (Highest vs Lowest) ---")
    logger.info(f"NA & ANZ Mean: {reg_high.mean():.4f} | Sub-Saharan Africa Mean: {reg_low.mean():.4f}")
    logger.info(f"Regional p-value: {p_reg:.4e}")
    
    reg_interp = "There is an extremely significant gap in well-being between these two regions."
    logger.info(f"Regional Interpretation: {reg_interp}")

    return {"pandemic_msg": interp}

# ===== Task 5: Correlation and Multiple Comparisons =====
@task
def correlation_multiple_comparisons(df):
    logger = get_run_logger()
    target = "happiness_score"  # Or "ladder_score"
    
    # 1. Identify explanatory variables
    exclude = [target, 'year', 'ranking']
    features = [col for col in df.select_dtypes(include=['number']).columns if col not in exclude]
    
    # 2. Setup thresholds
    alpha_original = 0.05
    num_tests = len(features)
    adjusted_alpha = alpha_original / num_tests
    
    logger.info(f"--- Task 5: Correlation Analysis (N={num_tests}) ---")
    logger.info(f"Original Alpha: {alpha_original} | Bonferroni Adjusted Alpha: {adjusted_alpha:.4f}")
    
    results = []
    for f in features:
        subset = df[[f, target]].dropna()
        
        if len(subset) > 2:
            coeff, p_val = stats.pearsonr(subset[f], subset[target])
            
            # --- New Logic for Explicit Logging ---
            is_sig_original = p_val < alpha_original
            is_sig_adj = p_val < adjusted_alpha
            
            # Updated Logger: explicitly flag both criteria
            logger.info(f"Feature: {f:25}")
            logger.info(f"  - r: {coeff:.3f} | p: {p_val:.2e}")
            logger.info(f"  - Significant at 0.05: {is_sig_original}")
            logger.info(f"  - Survives Bonferroni: {is_sig_adj}")
            
            results.append({
                'feature': f, 
                'r': coeff, 
                'sig_orig': is_sig_original,
                'sig_adj': is_sig_adj
            })

    # Find strongest significant predictor that survived the correction
    sig_only = [res for res in results if res['sig_adj']]
    
    if sig_only:
        strongest = max(sig_only, key=lambda x: abs(x['r']))
        logger.info(f"Strongest Robust Predictor: {strongest['feature']} (r={strongest['r']:.3f})")
        return strongest
    else:
        logger.info("No features survived the Bonferroni correction.")
        return {"feature": "None", "r": 0}


# ===== Task 6: Summary Report Implementation =====
@task
def generate_summary_report(df, stats_res, test_res, corr_res):
    logger = get_run_logger()
    logger.info("--- FINAL SUMMARY REPORT ---")

    # 1. Dataset Scope
    num_countries = df['country'].nunique()
    num_years = df['year'].nunique()
    logger.info(f"The merged dataset contains data from {num_countries} countries over {num_years} years (2015-2024).")

    # 2. Regional Rankings
    region_means = stats_res['region_mean']
    top_3 = region_means.head(3).index.tolist()
    bot_3 = region_means.tail(3).index.tolist()
    logger.info(f"Top 3 happiest regions: {', '.join(top_3)}")
    logger.info(f"Bottom 3 regions by happiness: {', '.join(bot_3)}")

    # 3. Pandemic Impact
    logger.info(f"Pandemic Impact Result: {test_res['pandemic_msg']}")

    # 4. Strongest Correlation
    # strongest_corr is the dictionary returned from Task 5
    feature = corr_res['feature']
    r_val = corr_res['r']
    logger.info(f"The variable most strongly correlated with happiness is '{feature}' (Pearson r = {r_val:.3f}), "
                "retaining its significance even after Bonferroni correction.")



@flow
def happiness_pipeline():
    base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/main/assignments/resources/happiness_project/world_happiness_"
    years = [str(y) for y in range(2015, 2025)]

    # Task 1
    df = load_all_happiness_data(base_url, years)

    # Task 2
    stats_results = descriptive_statistics(df)

    # Task 3
    create_visualizations(df)

    # Task 4
    test_results = run_hypothesis_tests(df)

    # Task 5: Correlation Analysis
    strongest_predictor = correlation_multiple_comparisons(df)

    # Task 6: Final Report
    generate_summary_report(df, stats_results, test_results, strongest_predictor )

if __name__ == "__main__":
    happiness_pipeline()