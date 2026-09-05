
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


# ML/LLM Question 2

# Task 1: Converting a date string like "2023-07-04" to day-of-week
# Answer: I would use deterministic code because standard programming languages handle date math perfectly, instantly, and for free.

# Task 2: Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# Answer: I would use an LLM because reading unstructured job descriptions requires language comprehension and human-like judgment.

# Task 3: Predicting customer churn given 15 numeric features and a labeled training dataset
# Answer: I would use a trained ML model because it excels at finding patterns in structured numeric columns and mapping them to a clear outcome.

# Task 4: Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# Answer: I would use an LLM because it natively understands that different words and short abbreviations can mean the exact same place.

# Task 5: Summing a column of revenue figures
# Answer: I would use deterministic code because standard math is perfectly accurate and fast, whereas LLMs are unreliable at arithmetic.
# ------------------------------------------------------------------------------


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

# Prompt Question 1

# Alternative System Prompt:
# "Write exactly two sentences recommending whether to go for a run today. The 
# first sentence must clearly state the final running prediction. The second 
# sentence must explain the specific weather reasoning behind it. Do not use 
# bullet points, headers, or conversational intros."
#
# What to change in the validation logic:
# 1. Update the sentence count check from `len(sentences) > 1` to `len(sentences) != 2`. 
#    This flags the output if it has too few or too many sentences.
# 2. Update the graceful fallback behavior. If the model fails the check, instead 
#    of keeping just the first sentence, the code should slice the first two 
#    sentences: `recommendation = ". ".join(sentences[:2]) + "."`.
# ------------------------------------------------------------------------------


# Prompt Question 2

import time

def call_with_retry(client, messages, max_retries=3):

    # Safely calls the OpenAI API and retries if a network or API error occurs.
    for attempt in range(max_retries):
        try:
            # Attempt the API call
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

# When you would use this in a production pipeline:
# You use retry logic because cloud networks and external APIs occasionally drop 
# requests, hit rate limits, or suffer brief outages. In an automated data pipeline, 
# you do not want one single network hiccup to crash your entire batch job 
# halfway through. This function acts as a safety shield, keeping the pipeline 
# moving forward smoothly during temporary API blips.
# ------------------------------------------------------------------------------
