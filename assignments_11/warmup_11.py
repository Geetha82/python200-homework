

from prefect import task, get_run_logger

# Prefect Orchestration

# Prefect Question 1

# What is the difference between a @flow and a @task?
# * A @flow is like the manager. It controls the whole pipeline, decides the order 
#   of steps, and brings everything together.
# * A @task is like the worker. It is a single, small step inside the flow that 
#   does one specific job (like loading data from an API or saving to a database).

# Would you decorate a simple Celsius-to-Fahrenheit function with @task?
# No, I would not. 

# Prefect tracks every single @task, creates logs for it, and checks its status. 
# This adds a tiny bit of extra work and slowdown for your computer. For a super quick, 
# simple math problem with no internet or database work, you do not need that extra tracking. 
# It is better to leave it as a regular Python function so it runs instantly.

# Prefect Question 2

@task(name="call_api", retries=3, retry_delay_seconds=30)
def call_api():
    pass

# Prefect Question 3

# Where in the UI do you look to see what went wrong?

# I  would click on the name of the specific failed flow run in the Prefect dashboard.
# Then, I would look at the "Logs" tab or click directly on the failed "transform" task box in the graph.

# What specific information would you expect to find there?

# * I expect to find the specific Python Error Message and the full Stack Trace 
#   (the lines of code showing exactly where the crash happened).
# * I would also see the Log Level messages (like ERROR or CRITICAL) showing the 
#   inputs or variables used right before the task crashed (for example, a network 
#   timeout or a missing key in the data).


# Production Patterns

# Production Question 1
# What does raise_for_status() do?
# It tells the code to stop immediately if the website or API sends back an error code (like an HTTP 500 error). 
# It raises an official Python error to stop the script.

# Why is it better than a print("error") statement?
# If print("error") is used, the code keeps running blindly even though it has no data. 
# If raise_for_status()is used, Prefect knows the step failed and will stop the script or 
# try running it again.

# What happens to downstream tasks when a 500 error hits?
# * With print("error"): The task passes empty or broken data to the next steps. 
#   The later steps will crash anyway with ugly errors like "KeyError".
# * With raise_for_status(): The pipeline halts instantly. The later steps are cleanly 
#   canceled by Prefect before they can break or cause damage.

# Production Question 2
# What does upsert protect you from in this scenario?
# It saves you from duplicate data errors. Because "load_raw" already ran before the crash, 
# those rows are already in your database. When you restart the script, upsert sees the 
# same "date" and simply updates the existing row instead of making a duplicate.

# What would happen if you used a plain insert instead?
# The database will reject the data and throw a primary key error because that "date" already 
# exists. The pipeline will crash at the "load_raw" step, and your fixed "transform" code 
# will never get a chance to run.

# Production Question 3
task(name="load_enriched_records")
def load_enriched(enrichment_records: list):
    get_run_logger().info(f"Successfully upserted {len(enrichment_records)} enrichment records.")

# Production Question 4
# How does the incremental processing check contribute to idempotency?
# An incremental check ensures your pipeline only processes *new* or *updated* rows 
# instead of re-running everything from scratch. It checks what dates are already inside 
# your database and skips them. This ensures that no matter how many times you run the 
# pipeline, it only does the actual transformation work for a day once.

# What are the practical consequences of removing it?
# 1. Cost: You would have to pay for OpenAI API tokens to re-generate LLM text for all 
#    365 days every single day. Your API bill would skyrocket unnecessarily.
# 2. Time: LLM API requests are slow. Waiting for 365 separate text generations every 
#    day would make a pipeline that should take 5 seconds take several minutes to finish.
# 3. Data Correctness: Since LLMs are creative, re-running them would overwrite your 
#    old, existing recommendations with brand new sentences every day. Your historical 
#    data would constantly shift instead of staying permanent and reliable.
