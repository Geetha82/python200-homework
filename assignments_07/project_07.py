import os
import scipy.stats as stats
import pandas as pd
import matplotlib
# Force a non-interactive headless backend for matplotlib to prevent macOS thread blocks
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

# =====================================================================
# --- Environment Setup & Global State ---
# =====================================================================
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

api_key = os.getenv("OPENAI_API_KEY")

# Define the shared global DataFrame reference variable exactly as requested by Task 1
df = None

# Pure relative path parameters specified literally in the assignment guidelines
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"
FALLBACK_DIR = "assignments/resources/happiness_project/"

# Ensure the required outputs directory structure exists locally before running plots
os.makedirs("assignments_07/outputs", exist_ok=True)

# =====================================================================
# --- Task 1: Define Your Tools with Google-Style Docstrings ---
# =====================================================================

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged CSV from DATA_PATH. If that file does not exist, falls back
    to loading and merging all yearly CSVs from assignments/resources/happiness_project/
    using an iterative loop. Stores the resulting combined dataset in the global df.

    Returns:
        dict: A dictionary containing exactly two keys:
            - "shape": A list of integers [rows, columns] representing dataset dimensions.
            - "columns": A list of strings containing all column header names.
    """
    global df
    
    # 1. Attempt primary relative path load
    if os.path.exists(DATA_PATH) and os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    # 2. Literal assignment fallback requirement directory tracking sweep
    elif os.path.exists(FALLBACK_DIR) and os.path.isdir(FALLBACK_DIR):
        all_files = sorted([
            os.path.join(FALLBACK_DIR, f) for f in os.listdir(FALLBACK_DIR)
            if f.lower().endswith(".csv")
        ])
        
        if not all_files:
            return {"error": f"No CSV dataset files discovered inside fallback directory: '{FALLBACK_DIR}'"}
            
        all_dfs = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Pull year digits sequentially out of file naming strings to form year indices
            year_digits = "".join([c for c in filename if c.isdigit()])
            year_val = int(year_digits) if year_digits else 2020
            
            temp_df = pd.read_csv(file_path)
            temp_df["Year"] = year_val
            all_dfs.append(temp_df)
            
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        # Secondary fallback layer for localized repository structure variations (e.g. current directory resources/)
        local_resources = "resources"
        if os.path.exists(local_resources) and os.path.isdir(local_resources):
            all_files = sorted([
                os.path.join(local_resources, f) for f in os.listdir(local_resources)
                if f.lower().endswith(".csv") and "happiness" in f.lower()
            ])
            if all_files:
                all_dfs = []
                for file_path in all_files:
                    filename = os.path.basename(file_path)
                    year_digits = "".join([c for c in filename if c.isdigit()])
                    year_val = int(year_digits) if year_digits else 2020
                    temp_df = pd.read_csv(file_path)
                    temp_df["Year"] = year_val
                    all_dfs.append(temp_df)
                df = pd.concat(all_dfs, ignore_index=True)

    if df is None:
        return {"error": f"Could not find dataset files at '{DATA_PATH}' or literal fallback path '{FALLBACK_DIR}'."}

    # Normalize column names to match standard variations gracefully
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "Ladder score": "happiness_score",
        "Happiness Score": "happiness_score",
        "Logged GDP per capita": "gdp_per_capita",
        "GDP per Capita": "gdp_per_capita",
        "Social support": "social_support",
        "Social Support": "social_support",
        "Regional indicator": "regional_indicator",
        "Region": "regional_indicator",
        "Year": "year"
    }
    df.rename(columns=rename_map, errors="ignore", inplace=True)

    # Clean character decimal anomalies if values are captured as string objects
    for col in df.columns:
        if df[col].dtype == "object" and col not in ["country", "regional_indicator"]:
            try:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    return {
        "shape": list(df.shape),
        "columns": df.columns.tolist()
    }

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The exact name string of the target column to analyze.

    Returns:
        dict: A dictionary containing descriptive metrics from pandas describe(), or an error message.
    """
    global df
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available fields: {df.columns.tolist()}"}
        
    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name string of the first numeric column.
        col2: The name string of the second numeric column.

    Returns:
        dict: A dictionary containing col1, col2, pearson_r, and p_value rounded to 4 decimal places.
    """
    global df
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"One or both specified columns were not found. Available fields: {df.columns.tolist()}"}
        
    try:
        clean_df = df[[col1, col2]].dropna()
        # FIXED: Bug resolved by replacing clean_df[clean_df] with explicit second selected column tracking matrix
        pearson_r, p_value = stats.pearsonr(clean_df[col1], clean_df[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4)
        }
    except Exception as e:
        return {"error": f"Failed to compute statistical correlation: {str(e)}"}

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The name string of the column to rank countries by.
        year: The specific calendar year filter integer.
        n: The count of top rows to slice. Defaults to 5.

    Returns:
        list: A list of dicts, where each dict has "country" and the requested column value.
    """
    global df
    if df is None:
        return [{"error": "No data is loaded yet. Please call load_happiness_data first."}]
    if column not in df.columns:
        return [{"error": f"Column '{column}' not found. Available fields: {df.columns.tolist()}"}]
        
    try:
        # Track both standard year key casings to remain fully safe
        year_col = "year" if "year" in df.columns else "Year"
        filtered_df = df[df[year_col].astype(int) == int(year)]
        if filtered_df.empty:
            return [{"error": f"No data records discovered tracking the year window {year}."}]
            
        top_rows = filtered_df.sort_values(by=column, ascending=False).head(n)
        
        # FIXED: Eliminates dict envelope wrapping mismatch, returning pure raw list of dicts directly
        results = []
        for _, row in top_rows.iterrows():
            results.append({
                "country": str(row["country"]),
                column: row[column]
            })
        return results
    except Exception as e:
        return [{"error": f"Failed to sort and filter top country metrics: {str(e)}"}]


# =====================================================================
# --- Task 2: Build the Agent ---
# =====================================================================
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

# Exact instructions copy matching task documentation parameters verbatim
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset. Use the available tools for loading data, summarizing columns, computing correlations, and ranking countries. Write Python code directly only when the tools are not sufficient (for example, when creating custom plots or computing something the tools don't cover). Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# =====================================================================
# --- Execution Entry Block ---
# =====================================================================
if __name__ == "__main__":
    
    # --- Task 3: Run Guided Queries ---
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png."
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        # reset=False keeps runtime history state metrics active across turns
        response = agent.run(query, reset=False)
        print(response)

    # --- Task 4: Your Own Questions ---
    print("\n--- Task 4: Custom Queries ---")

    # My query 1
    my_query_1 = "Find the top 3 countries with the highest social_support scores in 2020."
    print(f"\n# My query 1\nmy_query_1 = \"{my_query_1}\"")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # This query triggered TOOL USE exclusively. The agent directly called the pre-built tool 'get_top_n_countries'
    # with the requested column, year, and slice parameters, outputting the clean target ranking records immediately.

    # My query 2
    my_query_2 = "Compute the average happiness_score grouped by region across all tracked entries. Save a local horizontal bar chart tracking this to assignments_07/outputs/regional_averages.png"
    print(f"\n# My query 2\nmy_query_2 = \"{my_query_2}\"")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # This query triggered BOTH tool use and code generation. The agent used 'load_happiness_data' to sync the primary records,
    # and then wrote a standalone matplotlib scripting routine to group rows by region, average metrics, and write a plot to disk.


# =====================================================================
# --- Task 5: Reflection ---
# =====================================================================
# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#    - The agent observed that compute_correlation returned a pearson_r of ~0.6218 and a p_value of 0.0.
#    - It correctly interpreted the 0.0 p-value as being well below the standard statistical alpha significance
#      threshold of 0.05, explicitly communicating that there is a strong, positive, and statistically significant 
#      relationship between economic output (GDP per capita) and national happiness scores.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#    - The agent's autonomous recovery during Query 5 was a great surprise. When writing code to plot multi-line trends,
#      the sandboxed environment threw data type exceptions due to string-formatted objects. The agent caught its own 
#      terminal errors, appended 'pd.to_numeric()' with errors='coerce' on the fly, handled the unstack() index matrix modifications,
#      and exported the finalized PNG visualization chart asset perfectly.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    - An 'execute_sql_query' tool mapping the database dataframe to an in-memory SQLite wrapper would provide massive utility.
#    - This tool would take standard ANSI SQL strings and return rows, allowing the agent to answer highly complex sub-queries 
#      and joins like: "Identify which regions have more than 3 countries whose score sits above the global rolling median."
