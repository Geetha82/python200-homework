import pandas as pd
import numpy as np
from prefect import flow, task

# --- Prefect Pipeline Question 2 ---

@task
def create_series(arr):
    """Takes a NumPy array and returns a pandas Series named 'values'."""
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    """Removes any NaN values using .dropna()."""
    return series.dropna()

@task
def summarize_data(series):
    """Returns a dictionary with mean, median, std, and mode."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow(name="Warmup Summary Flow")
def pipeline_flow(arr):
    """Orchestrates the three functions as a Prefect flow."""
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

if __name__ == "__main__":
    # Input data from Q1
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    
    # Execute the flow
    results = pipeline_flow(arr)
    
    # Print results to confirm they match Q1
    print("\nFlow Results:")
    for key, val in results.items():
        print(f"{key}: {val:.2f}")

"""
--- Reflection ---

1. Why might Prefect be more overhead than it is worth here?
For a simple script with a small, static array, Prefect adds latency due to the orchestration 
engine (initializing the flow, tracking task states, and logging). The "home-grown" Python 
functions execute in microseconds, whereas Prefect adds a few seconds of setup time. It also 
requires the installation of an external library for logic that can be handled by standard Python.

2. Realistic scenarios where Prefect is useful even for simple logic:
- Scheduling: If this simple logic needs to run every Monday at 8:00 AM automatically.
- Observability/Alerting: If the pipeline fails (e.g., the data source is missing), Prefect 
  can send a Slack or Email notification without manual checking.
- Retries: If the 'data' comes from a flaky API, Prefect can automatically retry the task 
  3 times before failing.
- Infrastructure: If the simple logic needs to run inside a specific Docker container or 
  on a cloud worker, Prefect manages that environment deployment.
"""