
# ==============================================================================
# ML/LLM QUESTION 1: ROLE SWAPPING & MODEL CAPABILITIES
# ==============================================================================
# * Why the Classifier owns Binary Prediction:
#   The scikit-learn ML classifier is mathematically optimized for binary classification. 
#   By calculating a precise decision boundary across numerical data, it isolates tabular 
#   patterns to output a strict, deterministic matrix ([0] or) alongside an exact 
#   mathematical confidence score (via predict_proba) instantly and at near-zero cost.
#
# * Why the LLM owns Natural-Language Recommendations:
#   The LLM possesses deep semantic fluidness and structural linguistic comprehension. 
#   It excels at translating multiple context features into human prose, allowing it to 
#   synthesize both raw metrics and numerical model confidence into fluid, context-aware 
#   narrative summaries that traditional tabular models cannot construct.
#
# * What specifically breaks if swapped:
#   - If the LLM handles Classification: Downstream engineering stability completely breaks. 
#     LLMs are non-deterministic; a slight text variation or shift in phrasing (e.g., outputting 
#     "Yes, run!" or "Favorable conditions" instead of an absolute integer 0 or 1) will 
#     instantly violate strict database schemas and crash the automated ingestion pipeline. 
#     It also introduces extreme API latency and unnecessary operational costs.
#   - If the ML model handles Text Generation: The transform step fails entirely. Traditional 
#     classification frameworks (like logistic regression or random forests) are mathematically 
#     incapable of generating unstructured vocabulary or freeform strings, yielding zero text output.

# ==============================================================================
# ML/LLM QUESTION 2: TOOL SELECTION FOR SPECIFIC TASKS
# ==============================================================================
# * Task 1 (Date to Day-of-Week): Deterministic code. Standard datetime math handles 
#   calendar logic with 100% precision, zero runtime latency, and zero compute costs.
#
# * Task 2 (Job Posting Classification): LLM. Unstructured, freeform job descriptions 
#   require semantic text comprehension to distill nuance and intent into a thematic bucket.
#
# * Task 3 (Predicting Customer Churn): Trained ML model. It is designed to map historical, 
#   tabular user behavior tables directly to a clear binary prediction with a measurable 
#   confidence metric.
#
# * Task 4 (Normalizing City Names): Deterministic code (with an explicit caveat). If dealing 
#   with a known, bounded set of operational regions, a fixed programmatic dictionary lookup 
#   mapping variations ("NY", "NYC", "N.Y.C.") to a single string ("New York City") is preferred. 
#   Deterministic mapping ensures absolute validation control and zero token cost. However, if the 
#   input suffers from open-ended, highly ambiguous human typos or unstructured variations, 
#   an LLM is justified because its semantic anchoring handles linguistic noise that regex fails to catch.
#
# * Task 5 (Summing Revenue Figures): Deterministic code. Financial calculations require absolute, 
#   exact mathematical precision. LLMs can easily fail at basic arithmetic due to tokenization 
#   biases, making native code the only safe choice.


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

# ==============================================================================
# PROMPT DESIGN: QUESTION 1
# ==============================================================================
# Part 1: Alternative System Prompt
# ------------------------------------------------------------------------------
ALTERNATIVE_SYSTEM_PROMPT = """
You are a precise weather recommendation engine. Analyze the provided weather 
data and the machine learning model's prediction. Generate a running 
recommendation that is exactly two sentences long. The first sentence must 
state whether conditions are favorable or unfavorable based on the data. The 
second sentence must provide a brief explanation why, citing at least one 
specific weather feature. Do not include any other text, introductory phrases, 
or formatting.
"""

# Part 2: Validation Logic Shift Change
# ------------------------------------------------------------------------------
# To accommodate this alternative output format, the pipeline's verification boundaries 
# must shift from checking a strict binary threshold to a structural rule evaluation. 
# Specifically, the validation layer must be updated to change its parsing count constraint 
# from a single delimiter check to a multi-sentence boundary count. Instead of checking 
# for an exact one-sentence response, the code must split the LLM response string on sentence 
# delimiters (like periods, exclamation points, or question marks) and verify that it 
# contains exactly two structural segments. 
#
# If a violation is caught during processing (e.g., the LLM returns three sentences), 
# the defensive fallback logic must shift from trimming a single phrase to cleanly 
# isolating and splicing only the first two complete text statements. This ensures the 
# data satisfies your database schema constraints without dropping the primary evaluation 
# or crashing on edge cases.
# ==============================================================================


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


# When you would use this in a production pipeline:
# You use retry logic because cloud networks and external APIs occasionally drop 
# requests, hit rate limits, or suffer brief outages. In an automated data pipeline, 
# you do not want one single network hiccup to crash your entire batch job 
# halfway through. This function acts as a safety shield, keeping the pipeline 
# moving forward smoothly during temporary API blips.
# ------------------------------------------------------------------------------



