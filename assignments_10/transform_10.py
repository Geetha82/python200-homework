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
                sentences = [s for s in recommendation.split('.') if s.strip()]
                if len(sentences) > 1:
                    recommendation = sentences.strip() + "."
            else:
                print(f"API Error on date {record['date']}. Injecting generic fallback recommendation.")
                fallback_status = "favorable" if record["good_for_running"] == 1 else "unfavorable"
                recommendation = f"Weather conditions appear {fallback_status} for running based on algorithmic metrics."

            # 🎯 FIXED PAYLOAD: Strictly writing the four required core enrichment fields
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
#   - Why: The ML classifier had a high confidence score of 0.9994. The LLM successfully 
#     picked up on this certainty and correctly avoided any tone hedging.
#
# * Weaker Summary Example (Sample #2 - 2023-01-01):
#   - Text: "Avoid running today due to extremely unfavorable weather conditions."
#   - Why: While accurate, the sentence is far too generic and fails to explain *why* it is unfavorable.
#   - Cause: A low model temperature setting (0.3) combined with a strict one-sentence constraint 
#     forces gpt-4o-mini to write wide generalizations instead of parsing descriptive traits.
# ==============================================================================


# ==============================================================================
# STEP 6: CONCEPTUAL REFLECTION BLOCK (STREAMLINED & CLEAN)
# ==============================================================================
# 1. Geographic Generalization: No, the classifier will likely become inaccurate. It was 
#    trained specifically on Charlotte, NC climate data, so running it on a city with 
#    a radically different environment represents 'data drift' where input thresholds 
#    fall completely outside what the model learned during training.
#
# 2. LLM Capabilities & Constraints: The LLM has no ability to override the classifier; 
#    it is purely additive. Because the script feeds the classifier's choice directly into 
#    the prompt as truth, a faulty ML prediction means the LLM will simply write anarticulate, 
#    convincing justification defending that incorrect decision.
# 
# 3. Scaling Bottlenecks: The main concerns would be high cloud API transactional costs andnetwork 
#    latency from making 50,000 individual calls. This would be addressed by usingOpenAI's asynchronous 
#    Batch API for a 50% discount or hosting a local open-source model.
# ==============================================================================

if __name__ == "__main__":
    run_pipeline()