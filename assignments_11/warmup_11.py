# warmup_11.py

from prefect import task, get_run_logger

# PREFECT ORCHESTRATION

# --- Prefect Question 1 ---

# What is the difference between a @flow and a @task?
# * A @flow is the high-level manager that controls the overall pipeline, orchestrates 
#   the execution sequence, manages parameters, and links steps together.
# * A @task is a single, discrete unit of work inside a flow that handles a specific 
#   isolated job (like extraction, database manipulation, or transformations).

# Would you decorate a simple Celsius-to-Fahrenheit function with @task?
# No, I would not. 

# Prefect tracks every single @task, captures metadata, and logs its state to the server. 
# This process introduces overhead that slows down local processing. For a quick, in-memory 
# mathematical utility calculation with zero network or disk I/O, it is better to leave it 
# as a standard Python function so it executes instantly without extra tracking.


# --- Prefect Question 2 ---
@task(name="call_api", retries=3, retry_delay_seconds=30)
def call_api():
    pass

# --- Prefect Question 3 ---

# Where in the UI do you look to see what went wrong?
# * I would navigate to the Flow Runs tab on the Prefect UI, select the failed run instance, 
#   and look directly at the unified "Logs" stream tab.
# * Alternatively, I would go to the task run graph or grid view and click directly on the 
#   red "transform" task block to isolate its logs.

# What specific information would you expect to find there?
# * I would look for the detailed Python Exception type, error message, and a full 
#   traceback showing the exact file and line number where the code crashed.
# * I would also check any surrounding context logs (like input payload keys or missing parameters) 
#   written just before the task changed its state to Failed.


# PRODUCTION PATTERNS

# --- Production Question 1 ---

# What does raise_for_status() do?
# It is a built-in requests utility that explicitly checks the HTTP status code of an 
# outbound request response. If the server responds with a 4xx or 5xx error code, it raises 
# a clean HTTPError exception on the spot.

# Why is it better than a print("error") statement?
# Using print("error") handles the issue visually but leaves the pipeline script running. 
# The task finishes as "Completed" passing broken, empty data down the chain. Using 
# raise_for_status() triggers a real exception that signals Prefect that the task explicitly 
# failed, halting the pipeline or initiating retries.

# What happens to downstream tasks when a 500 error hits?
# * With print("error"): The task completes anyway. The empty or invalid data is passed 
#   downstream, causing subsequent tasks (like transform) to crash ungracefully with 
#   unrelated errors like KeyError or TypeError.
# * With raise_for_status(): The task crashes immediately. Prefect handles the exception, 
#   cancels all downstream tasks gracefully before they run with bad data, and flags the 
#   run as Failed.

# --- Production Question 2 ---

# What does upsert protect you from in this scenario?
# It protects you from database constraint errors and duplicate data generation on restarts. 
# Since the "extract" and "load_raw" steps already successfully loaded the rows to Supabase before 
# the transform crash, those entries exist in the database. When you restart the pipeline, 
# upsert matches the existing "date" keys and updates them safely without failing.

# What would happen if you used a plain insert instead?
# The pipeline would crash at the "load_raw" task on the second run. The database would check 
# the "date" column, find that those unique keys already exist, and throw a Unique Constraint 
# Violation error. This prevents your corrected transform task from ever executing.


# --- Production Question 3 ---
@task(name="load_enriched_records")
def load_enriched(enrichment_records: list):
    get_run_logger().info(f"Successfully upserted {len(enrichment_records)} enrichment records.")

# --- Production Question 4 ---

# How does the incremental processing check contribute to idempotency?
# An incremental check ensures that your pipeline verifies what data has already been fully 
# processed by looking up historical keys before committing computing cycles. By filtering 
# out active matches, it guarantees that re-running the pipeline multiple times processes 
# new rows exactly once, resulting in a safe data state.

# What are the practical consequences of removing it?
# 1. Cost: You would submit all 365 weather records to the OpenAI API on every execution, 
#    creating high token consumption costs and bloated utility bills.
# 2. Time: Making 365 separate external HTTP network requests to a generative LLM endpoint 
#    sequentially is slow. A pipeline that should take seconds will stall for minutes.
# 3. Data Correctness: Since LLM model outputs are inherently creative and non-deterministic, 
#    re-running them over the same history would overwrite old text with brand-new sentences, 
#    ruining historical data consistency.

