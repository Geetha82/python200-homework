# Video link - https://youtu.be/ba0TdgF-BiA

import os
import json
import time
import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# Load project credentials from your local .env file
load_dotenv()

# Initialize external database and API clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_with_retry(client, messages, max_retries=3):

    # Safely executes an OpenAI API request, retrying up to 3 times on network errors.
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3
            )
            return response
        except Exception as e:
            print(f" API attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    print("All retry attempts failed.")
    return None


def run_pipeline():
    # --------------------------------------------------------------------------
    # STEP 1: Incremental Read
    # --------------------------------------------------------------------------
    print("Initializing Step 1: Incremental Read...")
    
    metadata_path = "models/weather_classifier_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    FEATURES = metadata["feature_names"]
    
    raw_response = supabase.table("weather_raw").select("*").execute()
    all_raw_records = raw_response.data
    
    enriched_response = supabase.table("weather_enriched").select("date").execute()
    processed_dates = {row["date"] for row in enriched_response.data}
    
    unprocessed_records = [r for r in all_raw_records if r["date"] not in processed_dates]
    
    print(f"\n--- PROCESSING PIPELINE SUMMARY ---")
    print(f"Total raw records existing: {len(all_raw_records)}")
    print(f"Total already enriched:     {len(processed_dates)}")
    print(f"Records to process now:     {len(unprocessed_records)}")
    print(f"-----------------------------------\n")
    
    # --------------------------------------------------------------------------
    # STEPS 2-4: Run Transformations ONLY if new records exist
    # --------------------------------------------------------------------------
    if not unprocessed_records:
        print(" No new records to process. Skipping transformation steps.")
    else:
        # STEP 2: ML Transform
        print("Initializing Step 2: ML Transform...")
        clf = joblib.load("models/weather_classifier.pkl")
        
        df = pd.DataFrame(unprocessed_records)
        X_new = df[FEATURES]
        
        df["good_for_running"] = clf.predict(X_new)
        probabilities = clf.predict_proba(X_new)
        df["confidence"] = probabilities[:, 1]
        
        enrichment_records = []
        for _, row in df.iterrows():
            enrichment_records.append({
                "date": row["date"],
                "good_for_running": int(row["good_for_running"]),
                "confidence": float(row["confidence"]),
                "temperature_2m_max": float(row["temperature_2m_max"]),
                "temperature_2m_min": float(row["temperature_2m_min"]),
                "precipitation_sum": float(row["precipitation_sum"]),
                "wind_speed_10m_max": float(row["wind_speed_10m_max"])
            })
            
        total_good_days = df["good_for_running"].sum()
        print(f"--- ML INFERENCE SUMMARY ---")
        print(f"Days classified as good for running: {total_good_days} / {len(df)}")
        print(f"Confidence score range:              {df['confidence'].min():.4f} to {df['confidence'].max():.4f}")
        print(f"----------------------------\n")
        
        # STEP 3: LLM Transform
        print("Initializing Step 3: LLM Transform...")
        payloads = []
        
        system_prompt = """You are an automated backend step in an ETL database pipeline. 
Your single job is to output a direct, practical one-sentence running recommendation based on the data.
Strict Constraints:
1. Output exactly one sentence.
2. Do not use markdown quotes, headers, or bullet points.
3. Completely skip conversational greetings or intros.
4. Hedge your certainty tone if the pipeline confidence rating sits close to 50%."""

        for index, record in enumerate(enrichment_records, start=1):
            ml_label = "Good day for a run" if record["good_for_running"] == 1 else "Bad day for a run"
            confidence_pct = record["confidence"] * 100
            
            user_message = f"""Context metrics:
- Max Temp: {record['temperature_2m_max']}°F, Min Temp: {record['temperature_2m_min']}°F
- Precipitation: {record['precipitation_sum']} in, Max Wind: {record['wind_speed_10m_max']} mph
- Model Core Assessment: {ml_label}
- Pipeline Confidence Score: {confidence_pct:.1f}%"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = call_with_retry(openai_client, messages)
            
            if response is not None:
                recommendation = response.choices[0].message.content.strip()
                sentences = [s.strip() for s in recommendation.split('.') if s.strip()]
                if len(sentences) > 1:
                    recommendation = ". ".join(sentences[:1]) + "."

            else:
                print(f"API Error on date {record['date']}. Injecting generic fallback recommendation.")
                fallback_status = "favorable" if record["good_for_running"] == 1 else "unfavorable"
                recommendation = f"Weather conditions appear {fallback_status} for running based on algorithmic metrics."

            payloads.append({
                "date": record["date"],
                "good_for_running": record["good_for_running"],
                "confidence": record["confidence"],
                "llm_summary": recommendation
            })
            
            if index % 50 == 0:
                print(f"⏳ Pipeline Progress Check: Completed {index} / {len(enrichment_records)} records...")

        # STEP 4: Load
        print(f"\nInitializing Step 4: Load...")
        print(f"Upserting {len(payloads)} validated records to weather_enriched...")
        result = supabase.table("weather_enriched").upsert(payloads).execute()
        print(f"Successfully upserted {len(result.data)} rows into weather_enriched!")
        print("Pipeline write operations completed successfully.")

    # --------------------------------------------------------------------------
    # STEP 5: Verify
    # --------------------------------------------------------------------------
    print("\n--- STEP 5: VERIFY ---")
    
    verify_response = supabase.table("weather_enriched").select("date", "good_for_running", "confidence", "llm_summary").execute()
    
    total_rows = len(verify_response.data)
    print(f"Total number of rows in weather_enriched: {total_rows}")
    
    total_good_days_all = sum(1 for row in verify_response.data if row["good_for_running"] == 1)
    print(f"Total days classified as good for running: {total_good_days_all}")
    
    print("\nFive sample rows from weather_enriched:")
    sample_rows = verify_response.data[:5]
    for idx, row in enumerate(sample_rows, start=1):
        print(f"  Sample #{idx}:")
        print(f"    date:             {row['date']}")
        print(f"    good_for_running: {row['good_for_running']}")
        print(f"    confidence:       {row['confidence']:.4f}")
        print(f"    llm_summary:      {row['llm_summary']}\n")
    print("-----------------------\n")


# ==============================================================================
# STEP 5: MODEL COMPARISON & CRITIQUE
#
# Evaluation of LLM Summaries:
# Overall, the LLM summaries accurately translate the binary predictions and confidence
# metrics into clear human prose without breaking the schema. 
#
# Strong Summary Example (2026-09-02):
# The summary for 2026-09-02 is highly accurate because the ML model predicted 'True' 
# with an exceptionally high confidence score of 0.9994. The LLM correctly identified 
# this extreme certainty and generated a strong, definitive validation: "It is highly 
# recommended to go for a run today given the favorable weather conditions."
#
# Weaker/Imperfect Summary Example (2023-01-03):
# The summary for 2023-01-03 is weaker because the LLM focuses almost entirely on the 
# pipeline's metadata rather than the underlying weather features. Instead of describing 
# the actual temperature or wind conditions that caused a 'False' prediction, it explicitly 
# narrates its own low confidence ("Given the extremely low confidence score..."). This 
# likely happened because the prompt heavily prioritized the `confidence` variable (0.0067) 
# over the raw weather feature strings, causing the LLM to meta-analyze the score instead 
# of summarizing the physical environment.
# ==============================================================================


# ==============================================================================
# STEP 6: PIPELINE REFLECTION
#
# Training Data vs. New Geographies:
# Because the scikit-learn classifier was trained exclusively on historical data from 
# Charlotte, NC, its predictions will likely degrade in accuracy if fed weather data 
# from a fundamentally different climate zone. Machine learning models assume that the 
# training distribution mirrors the production distribution; if the new city experiences 
# extreme baselines—such as desert heat or high-altitude cold—the feature weights optimized 
# for Charlotte's mild climate will misclassify conditions.
#
# LLM Override Capabilities & Pipeline Implications:
# In this architecture, the LLM acts as a purely additive layer and possesses zero 
# authority to 'override' the core classifier's binary decision. It is explicitly fed 
# the `good_for_running` classification as an immutable truth constraint in its prompt, 
# meaning it cannot modify the 'True' or 'False' schema field stored in Supabase. The major 
# implication here is that if the upstream ML model makes a faulty prediction, the LLM 
# is forced to defensively rationalize that error in prose, potentially creating misleading 
# descriptions to justify an incorrect classification.
#
# Scaling Concerns (50,000 Records):
# Scaling this pipeline from 366 records to 50,000 records makes API latency and compounding 
# financial costs my primary engineering concerns. While the scikit-learn classifier evaluates 
# 50,000 rows locally in milliseconds, making 50,000 sequential chat completion requests 
# to OpenAI would take hours to complete and incur substantial token expenses. To address 
# these bottlenecks, I would refactor the LLM transform layer to use asynchronous batch API 
# requests (processing records concurrently) and implement a caching layer to completely skip 
# LLM generation for identical weather profiles.
# ==============================================================================

if __name__ == "__main__":
    run_pipeline()