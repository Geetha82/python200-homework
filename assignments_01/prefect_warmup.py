import numpy as np
import pandas as pd
from prefect import task, flow

# Data from Pipeline Question 1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task(name="Create Series")
def create_series(arr):
    """Converts NumPy array to a pandas Series named 'values'."""
    return pd.Series(arr, name="values")

@task(name="Clean Data")
def clean_data(series):
    """Removes NaN values from the Series."""
    return series.dropna()

@task(name="Summarize Data")
def summarize_data(series):
    # Avoid adding complex type hints like ': pd.Series' for now
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "mode": float(series.mode()[0]) # Get single value as per Q1 instructions
    }

@flow(name="Data Processing Pipeline")
def pipeline_flow(arr):
    """Prefect Flow orchestrating the data tasks."""
    s = create_series(arr)
    cleaned = clean_data(s)
    results = summarize_data(cleaned)
    
    # Print results inside the flow for visibility
    print("\n--- Prefect Pipeline Results ---")
    for key, value in results.items():
        print(f"{key.capitalize()}: {value:.2f}")
    
    return results

if __name__ == "__main__":
    # Execute the flow
    pipeline_flow(arr)

# ===== Reflection Questions =====
"""
1. Why might Prefect be more overhead than it is worth here?
Prefect adds significant boilerplate (decorators, environment setup, and tracking logic) 
for a process that runs in milliseconds on a local machine. For a simple script with 
no external dependencies or long-running steps, plain Python is faster to write 
and easier to debug without the added layer of an orchestration engine.

2. Describe scenarios where Prefect is useful, even for simple logic:
- Scheduling: If this simple pipeline needed to run every Monday at 8:00 AM automatically.
- Retries: If the 'create_series' step involved fetching data from a flaky API that 
  fails 10% of the time, Prefect can automatically retry it.
- Observability: If we needed a dashboard (Prefect Cloud) to see exactly when the 
  pipeline ran, if it succeeded, and how long each step took.
- Notifications: Sending a Slack or email alert immediately if the 'clean_data' 
  step fails due to unexpected data types.
"""
