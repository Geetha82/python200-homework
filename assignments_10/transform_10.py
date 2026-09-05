
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

# Initialize external infrastructure clients
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
            print(f"API attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    print("All retry attempts failed.")
    return None


def run_pipeline():
    
    # STEP 1: Incremental Read 

    print("Initializing Step 1: Incremental Read...")
    
    # 1. Load model metadata to retrieve feature ordering rules
    metadata_path = "models/weather_classifier_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    FEATURES = metadata["feature_names"]  # Adjusted to match your exact JSON key name
    
    # 2. Fetch all historical rows from weather_raw
    raw_response = supabase.table("weather_raw").select("*").execute()
    all_raw_records = raw_response.data
    
    # 3. Fetch all dates already present in weather_enriched
    enriched_response = supabase.table("weather_enriched").select("date").execute()
    processed_dates = {row["date"] for row in enriched_response.data}
    
    # 4. Filter down to records that still need transformation processing
    unprocessed_records = [r for r in all_raw_records if r["date"] not in processed_dates]
    
    # 5. Print out the step summary overview
    print(f"\n--- PROCESSING PIPELINE SUMMARY ---")
    print(f"Total raw records existing: {len(all_raw_records)}")
    print(f"Total already enriched:     {len(processed_dates)}")
    print(f"Records to process now:     {len(unprocessed_records)}")
    print(f"-----------------------------------\n")
    
    if not unprocessed_records:
        print(" No new records to process. weather_enriched table is fully up to date.")
    else:   

    # STEP 2: ML Transform 
        print("Initializing Step 2: ML Transform...")
    
        # 1. Load serialized Week 4 scikit-learn model pipeline
        clf = joblib.load("models/weather_classifier.pkl")
    
        # 2. Build DataFrame from the unprocessed records, ordering columns by metadata
        df = pd.DataFrame(unprocessed_records)
        X_new = df[FEATURES]
    
        # 3. Run predictions and extract target class probabilities
        df["good_for_running"] = clf.predict(X_new)
        probabilities = clf.predict_proba(X_new)
        df["confidence"] = probabilities[:, 1]  # Extract Class 1 (Good Running Day)
    
        # 4. Build structural base list of enrichment records with required parameters
        enrichment_records = []
        for _, row in df.iterrows():
            enrichment_records.append({
                "date": row["date"],
                "good_for_running": int(row["good_for_running"]),
                "confidence": float(row["confidence"]),
                # Preserve raw weather attributes needed downstream for the LLM
                "temperature_2m_max": float(row["temperature_2m_max"]),
                "temperature_2m_min": float(row["temperature_2m_min"]),
                "precipitation_sum": float(row["precipitation_sum"]),
                "wind_speed_10m_max": float(row["wind_speed_10m_max"])
            })
        
        # 5. Compute summary metrics and analytics across current inference window
        total_good_days = df["good_for_running"].sum()
        min_conf = df["confidence"].min()
        max_conf = df["confidence"].max()
        
        print(f"--- ML INFERENCE SUMMARY ---")
        print(f"Days classified as good for running: {total_good_days} / {len(df)}")
        print(f"Confidence score range:              {min_conf:.4f} to {max_conf:.4f}")
        print(f"----------------------------\n")
        
        # Step 3: LLM Transform
        print("Initializing Step 3: LLM Transform...")
        payloads = []
        
        # Static systemic rule set matching batch processing principles
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
            
            # User message passing features dynamically
            user_message = f"""Context metrics:
Max Temp: {record['temperature_2m_max']}°F, Min Temp: {record['temperature_2m_min']}°F
Precipitation: {record['precipitation_sum']} in, Max Wind: {record['wind_speed_10m_max']} mph
Model Core Assessment: {ml_label}
Pipeline Confidence Score: {confidence_pct:.1f}%"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Call API using retry safety layer
            response = call_with_retry(openai_client, messages)
        
            if response is not None:
                recommendation = response.choices[0].message.content.strip()  # Handled index bracket fix
                
                # Sentence count structural length validation
                sentences = [s for s in recommendation.split('.') if s.strip()]
                if len(sentences) > 1:
                    recommendation = sentences[0].strip() + "."
            else:
                # Graceful fallback string mapping on repeated API errors
                print(f"API Error on date {record['date']}. Injecting generic fallback recommendation.")
                fallback_status = "favorable" if record["good_for_running"] == 1 else "unfavorable"
                recommendation = f"Weather conditions appear {fallback_status} for running based on algorithmic metrics."

            # Map complete payload fields directly to database columns
            payloads.append({
                "date": record["date"],
                "temperature_2m_max": record["temperature_2m_max"],
                "temperature_2m_min": record["temperature_2m_min"],
                "precipitation_sum": record["precipitation_sum"],
                "wind_speed_10m_max": record["wind_speed_10m_max"],
                
                "good_for_running": record["good_for_running"],  
                "confidence": record["confidence"],              
                "llm_summary": recommendation                    
            })
            
            # Monitor execution progress updates every 50 records
            if index % 50 == 0:
                print(f"⏳ Pipeline Progress Check: Completed {index} / {len(enrichment_records)} records...")


        # STEP 4: Load

        if payloads:
            print(f"\nInitializing Step 4: Load...")
            print(f"Upserting {len(payloads)} validated records to weather_enriched...")
            
            # Execute batch payload injection
            result = supabase.table("weather_enriched").upsert(payloads).execute()
            
            # Print out write metrics confirmation log
            print(f" Successfully upserted {len(result.data)} rows into weather_enriched!")
            print("Pipeline write operations completed successfully.")
        else:
            print("No payload updates were packaged during execution pipeline loop.")
            return

    # STEP 5: Verify
    print("\n--- STEP 5: VERIFY ---")
        
    # Query database records to verify state
    verify_data = supabase.table("weather_enriched").select("date", "good_for_running", "confidence", "llm_summary").execute()
        
    # 1. Print total rows found
    total_rows = len(verify_data.data)
    print(f"Total number of rows in weather_enriched: {total_rows}")
        
    # 2. Print historical calculation aggregates
    total_good_days_all = sum(1 for row in verify_data.data if row["good_for_running"] == 1)  
    print(f"Total days classified as good for running: {total_good_days_all}")
        
    # 3. Print 5 sample rows showing requested parameters
    print("\nFive sample rows from weather_enriched:")
    sample_rows = verify_data.data[:5]  # Added .data here
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
#   - Why: The ML classifier had an extremely high confidence score of 0.9994. The LLM 
#     successfully picked up on this certainty and correctly avoided any hedging, generating 
#     a highly definitive, descriptive, and encouraging recommendation.
#
# * Weaker Summary Example (Sample #2 - 2023-01-01):
#   - Text: "Avoid running today due to extremely unfavorable weather conditions."
#   - Why: While accurate (the classifier predicted 0 with a 0.0000 confidence score), the 
#     sentence is far too generic. It fails to tell the user *why* it is unfavorable.
#   - Cause: This weaker summary happens because a low model temperature setting (0.3) 
#     combined with a strict one-sentence length constraint forces gpt-4o-mini to save token 
#     space by defaulting to safe, wide generalizations instead of parsing descriptive traits.
# ==============================================================================


# ==============================================================================
# STEP 6: CONCEPTUAL REFLECTION BLOCK
# ==============================================================================
# 1. City Generalization: No, the classifier will likely become inaccurate. It was 
#    trained specifically on Charlotte, NC climate thresholds, so running it on a 
#    radically different environment (like Phoenix or Minneapolis) represents "data drift" 
#    where new input metrics fall completely outside what the model learned.
#
# 2. LLM Override: The LLM has no ability to override the classifier; it is purely 
#    additive. Because the script feeds the classifier's choice (0 or 1) directly 
#    into the prompt as truth, a faulty ML prediction means the LLM will simply 
#    write a highly articulate and convincing lie defending an incorrect decision.
#
# 3. Scaling to 50,000 Records: The main concerns would be high financial costs and 
#    network latency from making 50,000 individual, slow API calls. I would fix this 
#    by using OpenAI's Batch API to process rows asynchronously at a 50% discount, 
#    or host a small open-source language model locally to keep data processing free.
# ==============================================================================


if __name__ == "__main__":
    run_pipeline()