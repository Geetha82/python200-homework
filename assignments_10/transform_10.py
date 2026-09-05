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
# STEP 5: COMMENT SUMMARY REFLECTION (REAL DATA CRITIQUE)
# ==============================================================================
# Looking closely at our actual printed terminal sample rows, the LLM summaries 
# successfully and accurately reflect the underlying weather metrics and predictions.
#
# * Particularly Good Summary Example (Sample #1 - 2026-09-02):
#   - Text: "It is an excellent day for a run with favorable temperatures and no precipitation."
#   - Critique: This summary is strong because the machine learning model had an extremely high 
#     certainty score of 0.9994. The LLM accurately recognized this context, skipped conversational 
#     fluff, and matched the high confidence with definitive, action-oriented encouragement.
#
# * Weaker Summary Example (Sample #2 - 2023-01-01):
#   - Text: "Avoid running today due to extremely unfavorable weather conditions."
#   - Critique: While this output is factually accurate, it represents a weaker recommendation 
#     because it fails to describe which specific features caused the issue. The low model 
#     temperature setting (0.3) mixed with strict sentence length limits forced gpt-4o-mini to save 
#     token space by choosing broad generalizations over concrete variables like rain or wind.
# ==============================================================================


# ==============================================================================
# STEP 6: CONCEPTUAL REFLECTION BLOCK 
# ==============================================================================
# 1.Geographic Generalization and Climate Drift:If the pipeline processes data from a city with a completely different 
#   climatethan Charlotte, NC, the classifier’s predictions will likely be highly inaccurate.
#   Traditional machine learning models are fundamentally bound to the statisticaldistributions of their training datasets. 
#   Because your Week 4 model was trainedon Charlotte's specific environmental 
#   thresholds—such as its baseline humidity,moderate wind spikes, and seasonal
#   temperature swings—it will not generalize wellto a dry desert like Phoenix or a 
#   sub-zero winter environment like Minneapolis.Feeding data from a radically 
#   different geography into this model represents aclassic 'data drift' failure,
#   where the mathematical boundaries established duringtraining no longer apply 
#   to the incoming production features.
# 
# 2.LLM Capabilities, Constraints, and Overrides:The LLM functions strictly as a downstream additive 
#   translation layer and has zeroarchitectural mechanism to 'override' the classifier's primary 
#   binary prediction.Because the pipeline code feeds the classifier's outcome (0 or 1) directly into 
#   thesystem prompt as ground-truth context, the LLM is logically trapped and forced togenerate text 
#   that justifies that specific classification. This architecture carriesa significant risk of creating a 
#   'hallucination wrapper' or a 'convincing lie.'If the machine learning model makes a major prediction error, 
#   the downstream LLMwill articulately write a highly believable, human-sounding sentence defending thatfaulty decision,
#   masking serious bugs in the pipeline from the end-user.
# 
# 3.Scaling Bottlenecks to 50,000 Records:If this ETL data pipeline were scaled up to 50,000 records, 
#   the primary engineeringconcerns would split between network latency and API transactional costs.
#   While your scikit-learn classifier can complete 50,000 numeric calculations 
#   locallyin a fraction of a second for essentially zero cost, making 50,000 individual,
#   sequential network API requests to OpenAI would take hours to execute and result inmassive usage bills. 
#   To remediate this issue in a production data warehouse, 
#   I wouldreplace the sequential loop with OpenAI's asynchronous Batch API, which 
#   processeslarge-scale background data tasks with a 50% discount.     
#   Alternatively, for completedata isolation and zero external costs,
#   I would swap out the proprietary API callfor a small, open-source model (like a quantized Llama model) hosted locally rightalongside the Supabase infrastructure.
# ==============================================================================

if __name__ == "__main__":
    run_pipeline()