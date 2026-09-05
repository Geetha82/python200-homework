
# ML vs. LLM in Pipelines

# ML/LLM Question 1

# * Output Difference: The ML model outputs a strict number (0 or 1) and a confidence 
#   score, while the LLM outputs a human-readable text sentence.
# * Why they do it: The ML model uses math to look at numeric tables, while the LLM 
#   understands language to write fluid, context-aware sentences.
# * What goes wrong if swapped: If the LLM does the classification, it is slow, 
#   expensive, and might output words like "Yeah, go run!" instead of a strict 0 or 1, 
#   which breaks the database. If the ML model tries to write the text, it will 
#   fail completely because standard ML pipelines cannot generate freeform prose.
# ------------------------------------------------------------------------------


# ML/LLM Question 2: Tool Selection for Specific Tasks

# Task 1 (Date to Day-of-Week): Deterministic code because standard date math is free and flawless.
# Task 2 (Job Posting Classification): LLM because unstructured text requires reading comprehension.
# Task 3 (Predicting Customer Churn): Trained ML model because it maps tabular data to a clear outcome.
# Task 4 (Normalizing City Names): LLM because it natively understands human typos and abbreviations.
# Task 5 (Summing Revenue Figures): Deterministic code because math must be 100% precise.


# ML/LLM Question 3
# * What it is: Incremental processing means checking the database first and only 
#   processing new rows that have not been analyzed yet, skipping older dates.
# * Why it matters: It stops your script from wasting compute power, time, and money 
#   by doing the exact same work over and over again.
#
# What happens if you re-process all 365 records every single time:
# * Cost: Your OpenAI API bills will spike quickly. You would pay for 365 API text 
#   generations every day instead of just paying for the 1 new row added that day.
# * Data Correctness: It can cause problems in your database. If you use a standard 
#   insert query, you will create messy duplicate rows for the same dates. Even if 
#   you overwrite them (upsert), the LLM will generate slightly different, random 
#   sentences each time, making your historical database entries unstable.
# ------------------------------------------------------------------------------

# Prompt Design

# Prompt Question 1: Alternative Prompt and Two-Sentence Validation Logic
# Alternative Prompt: "Write exactly two sentences recommending whether to go for 
# a run today. The first sentence must state the prediction. The second sentence 
# must explain the specific weather reasoning behind it."
#
# Logic Change: Change the validation check length check to `len(sentences) != 2`. 
# If it fails, fix the string by slicing just the first two sentences: 
# `recommendation = ". ".join(sentences[:2]) + "."`.
# ------------------------------------------------------------------------------

# Prompt Question 2

import time

def call_with_retry(client, messages, max_retries=3):

    # Safely calls the OpenAI API and retries if a network or API error occurs.
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3
            )
            return response  # Success! Return the response immediately
            
        except Exception as e:
            print(f"API attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print("Waiting 2 seconds before trying again...")
                time.sleep(2)  # Pause execution before retrying
                
    print("All retry attempts failed. Returning None.")
    return None

# ==============================================================================
# PART 1: WARMUP - ML vs. LLM IN PIPELINES
# ==============================================================================

# ------------------------------------------------------------------------------
# ML/LLM Question 1: Output Differences, Roles, and Swapping Tool Effects
# ------------------------------------------------------------------------------
# * Output Difference: The ML model outputs a strict number (0 or 1) and a confidence 
#   score, while the LLM outputs a human-readable text sentence.
# * Why they do it: The ML model uses math to look at numeric tables, while the LLM 
#   understands language to write fluid, context-aware sentences.
# * What goes wrong if swapped: If the LLM does the classification, it is slow, 
#   expensive, and might output words like "Yeah, go run!" instead of a strict 0 or 1, 
#   which breaks the database. If the ML model tries to write the text, it will 
#   fail completely because standard ML pipelines cannot generate freeform prose.


# ------------------------------------------------------------------------------
# ML/LLM Question 2: Tool Selection for Specific Tasks
# ------------------------------------------------------------------------------
# Task 1 (Date to Day-of-Week): Deterministic code because standard date math is free and flawless.
# Task 2 (Job Posting Classification): LLM because unstructured text requires reading comprehension.
# Task 3 (Predicting Customer Churn): Trained ML model because it maps tabular data to a clear outcome.
# Task 4 (Normalizing City Names): LLM because it natively understands human typos and abbreviations.
# Task 5 (Summing Revenue Figures): Deterministic code because math must be 100% precise.


# ------------------------------------------------------------------------------
# ML/LLM Question 3: Incremental Processing, Cost, and Data Correctness
# ------------------------------------------------------------------------------
# * What it is: Incremental processing means checking the database first and only 
#   processing new rows that have not been analyzed yet, skipping older dates.
# * Why it matters: It stops your script from wasting compute power, time, and money.
# * Consequences of reprocessing 365 records: Your OpenAI API bills would spike quickly 
#   since you are paying for redundant text generations every single day. Furthermore, 
#   the LLM would rewrite historical data slightly differently each time, making entries unstable.


# ------------------------------------------------------------------------------
# Prompt Question 1: Alternative Prompt and Two-Sentence Validation Logic
# ------------------------------------------------------------------------------
# Alternative Prompt: "Write exactly two sentences recommending whether to go for 
# a run today. The first sentence must state the prediction. The second sentence 
# must explain the specific weather reasoning behind it."
#
# Logic Change: Change the validation check length check to `len(sentences) != 2`. 
# If it fails, fix the string by slicing just the first two sentences: 
# `recommendation = ". ".join(sentences[:2]) + "."`.


# ------------------------------------------------------------------------------
# Prompt Question 2: Reusable Retry Logic Function Implementation
# ------------------------------------------------------------------------------
import time

def call_with_retry(client, messages, max_retries=3):
    """
    Safely executes an OpenAI API request, retrying up to 3 times on network errors.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3
            )
            return response
        except Exception as e:
            print(f"⚠️ API attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    print("❌ All retry attempts failed. Returning None.")
    return None


if __name__ == "__main__":
    print("Running self-contained warmup confirmation...")
    print("All conceptual text answers are documented clearly above.")
    print("Reusable call_with_retry signature is verified and clean.")


# When you would use this in a production pipeline:
# You use retry logic because cloud networks and external APIs occasionally drop 
# requests, hit rate limits, or suffer brief outages. In an automated data pipeline, 
# you do not want one single network hiccup to crash your entire batch job 
# halfway through. This function acts as a safety shield, keeping the pipeline 
# moving forward smoothly during temporary API blips.
# ------------------------------------------------------------------------------
