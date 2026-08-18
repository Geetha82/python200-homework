import os
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import matplotlib
# Enforce a non-interactive headless backend for matplotlib to prevent system thread blocks
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

# --- Environment Setup & Global State ---
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

api_key = os.getenv("OPENAI_API_KEY")

# Define the shared global DataFrame reference variable as required by Task 1
df = None

# Updated assignment relative path parameters matching your specified location tracking file
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"
FALLBACK_DIR = "assignments/resources/happiness_project/"

# Ensure output directories exist relative to the running workspace context
os.makedirs("outputs", exist_ok=True)
os.makedirs("assignments_07/outputs", exist_ok=True)

# =====================================================================
# --- Task 1: Define Your Tools with Google-Style Docstrings ---
# =====================================================================

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Attempts to load a pre-merged CSV from DATA_PATH. If that file does not exist,
    falls back to loading and merging all yearly CSV files from the fallback
    resources folder sequentially. Updates the shared global DataFrame variable 'df'.
    
    CRITICAL: This tool returns a metadata summary DICTIONARY containing 'shape' and 
    'columns' keys. It does NOT return a pandas DataFrame object.

    Returns:
        dict: A dictionary containing exactly two metadata keys:
            - 'shape': A list of integers [rows, columns] representing dataset dimensions.
            - 'columns': A list of strings containing all column header names.
    """
    global df
    
    # Resolve the requested relative path context flexibly based on execution directory
    path = Path(DATA_PATH)
    if not path.exists():
        # Fallback check if executing directly from inside the assignments_07 subfolder
        path = Path("..") / DATA_PATH
        
    if path.exists():
        df = pd.read_csv(path)
        print(f"[Tool Log]: Successfully loaded dataset from relative path: {path}")
    else:
        # Resolve dynamic fallback folder paths for raw yearly files
        fallback_path = Path(FALLBACK_DIR)
        if not fallback_path.exists():
            fallback_path = Path("..") / FALLBACK_DIR
            
        if fallback_path.exists() and any(fallback_path.glob("*.csv")):
            print(f"[Tool Log]: Master file not found. Loading and combining files from {fallback_path}...")
            csv_files = sorted([p for p in fallback_path.iterdir() if p.suffix.lower() == ".csv"])
            all_dfs = []
            for file_path in csv_files:
                year_digits = "".join([c for c in file_path.stem if c.isdigit()])
                year_val = int(year_digits) if year_digits else 2020
                
                temp_df = pd.read_csv(file_path)
                temp_df["Year"] = year_val
                all_dfs.append(temp_df)
                
            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
                print("[Tool Log]: Dataset successfully merged into global memory state.")
                
    if df is None:
        return {"error": f"Could not find dataset files at relative path '{DATA_PATH}' or fallback location."}

    # Standardize column names to clean lowercase snake_case for seamless agent compatibility
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "ladder_score" in df.columns and "happiness_score" not in df.columns:
        df = df.rename(columns={"ladder_score": "happiness_score"})

    # Standardize commas to dots in numeric columns and cast types cleanly
    for col in df.columns:
        if df[col].dtype == "object" and col not in ["country", "regional_indicator"]:
            try:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
                
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2020).astype(int)

    # FRAMEWORK STATE INJECTION: Enforce strict column standardization rules right inside the 
    # injected sandbox memory scope. This ensures that any dynamically written python code 
    # referencing 'df' will successfully match the clean snake_case columns, eliminating KeyErrors!
    try:
        if 'agent' in globals() or 'agent' in locals():
            agent.state["df"] = df
        elif idx_tracker := [obj for obj in globals().values() if isinstance(obj, CodeAgent)]:
            idx_tracker[0].state["df"] = df
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
        column: The name of the target column string to describe.

    Returns:
        dict: A dictionary of summary metrics from pandas.describe(), or an error message.
    """
    global df
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}
    
    clean_col = column.strip().lower().replace(" ", "_")
    if clean_col not in df.columns:
        return {"error": f"Column '{column}' not found. Available options: {df.columns.tolist()}"}
        
    return df[clean_col].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: Name of the first metric column string.
        col2: Name of the second metric column string.

    Returns:
        dict: A dictionary with keys col1, col2, pearson_r, and p_value rounded to 4 decimal places.
    """
    global df
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}
        
    clean_col1 = col1.strip().lower().replace(" ", "_")
    clean_col2 = col2.strip().lower().replace(" ", "_")
    
    if clean_col1 not in df.columns or clean_col2 not in df.columns:
        return {"error": f"One or both specified column header names were missing. Options: {df.columns.tolist()}"}
        
    try:
        clean_data = df[[clean_col1, clean_col2]].dropna()
        r_val, p_val = stats.pearsonr(clean_data[clean_col1], clean_data[clean_col2])
        return {
            "col1": clean_col1,
            "col2": clean_col2,
            "pearson_r": round(float(r_val), 4),
            "p_value": round(float(p_val), 4)
        }
    except Exception as e:
        return {"error": f"Statistical operation failed on inputs: {str(e)}"}

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The name of the column to rank by (e.g., 'Happiness score').
        year: The specific reporting year filter integer.
        n: The number of top rows to return. Defaults to 5.

    Returns:
        dict: A dictionary containing a list of top ranked country records, or an error dictionary.
    """
    global df
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}
        
    clean_col = column.strip().lower().replace(" ", "_")
    if clean_col not in df.columns:
        return {"error": f"Column '{column}' not found. Options: {df.columns.tolist()}"}
    if "year" not in df.columns:
        return {"error": "Year column not found in the dataset tracking index."}
    if "country" not in df.columns:
        return {"error": "Country column not found in the dataset tracking index."}
        
    try:
        year_mask = df["year"].astype(int) == int(year)
        filtered_df = df[year_mask]
        
        if filtered_df.empty:
            return {"error": f"No records found matching year {year}."}
            
        top_rows = filtered_df.sort_values(by=clean_col, ascending=False).head(n)
        results = []
        for _, row in top_rows.iterrows():
            results.append({
                "country": str(row["country"]),
                column: row[clean_col]
            })
        return {"top_countries": results}
    except Exception as e:
        return {"error": f"Ranking filtration failed: {str(e)}"}


# =====================================================================
# --- Task 2: Build the Agent ---
# =====================================================================

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations, and ranking countries.
Write Python code directly only when the tools are not sufficient (for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# =====================================================================
# --- Execution Orchestration Block ---
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
        response = agent.run(query, reset=False)
        print(response)

    # --- Task 4: Your Own Questions ---
    print("\n--- Task 4: Custom Queries ---")

    # My query 1
    my_query_1 = "Find the top 3 countries with the highest Social support scores in 2020 using the rank ranking tool."
    print(f"\n--- My Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # This query triggered TOOL USE primarily. The agent parsed the inputs directly into parameters for the 
    # 'get_top_n_countries' tool and outputted the dictionary metadata result without needing to compile custom code.

    # My query 2
    my_query_2 = "What is the unique list of years present in this dataset? Write a python script to inspect the columns."
    print(f"\n--- My Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # This query triggered CODE GENERATION. Because no pre-built atomic tool extracts uniquely sorted tracking lists 
    # for columns, the agent utilized its unique code-generation capability to extract the values from the underlying dataframe.


# =====================================================================
# --- Task 5: Reflection ---
# =====================================================================
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#    - The agent observed the output from compute_correlation, which returned a p-value of 0.0.
#    - It communicated significance explicitly, stating that because the p-value is close to zero, the positive 
#      relationship between GDP and happiness is highly significant and unlikely to occur by random chance.
#    - It correctly applied the standard alpha significance threshold of 0.05.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#    - The agent's capability to recover from data types bottlenecks in Query 5 was a significant surprise. 
#      When it first attempted to group and plot the data frames directly, it handled parsing anomalies by 
#      independently inspecting data arrays, setting line markers dynamically, and arranging subplots via 
#      unstack() commands natively inside its code generation loop.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    - A 'query_data_profile' tool would make the agent more useful.
#    - This tool would accept a column name string, look up its data type, and return the total null/missing count 
#      along with any text-to-numeric format warning exceptions present in the dataset.
#    - It would prevent the agent from writing trial-and-error pandas parsing loops when handling unseen data, 
#      helping it instantly answer structural questions like: "Which tracking metrics contain high missing value records?"
