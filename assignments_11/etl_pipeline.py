# Video link - https://youtu.be/pgyUJXDMHk4

import os
import json
import joblib
import requests 
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from prefect import task, flow, get_run_logger

# Load project credentials from your local .env file
load_dotenv()

# Initialize external database and API clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# extract task
@task(retries=2, retry_delay_seconds=10)
def extract() -> list:

    logger = get_run_logger()
    logger.info("Starting historical extraction for San Francisco (SFO)...")
    
    # San Francisco (SFO) Coordinates: Latitude 37.7749, Longitude -122.4194
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=37.7749&longitude=-122.4194"
        "&start_date=2023-01-01&end_date=2023-12-31"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
        "&timezone=auto"
    )
    
    response = requests.get(url)
    response.raise_for_status()
 
    
    api_data = response.json()
    daily = api_data.get("daily", {})
    
    # Extract columnar lists from the JSON payload
    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precipitation = daily.get("precipitation_sum", [])
    wind_speed = daily.get("wind_speed_10m_max", [])
    
    # Convert columnar lists into a beginner-friendly list of row dictionaries
    row_records = []
    for i in range(len(dates)):
        row_records.append({
            "date": dates[i],
            "temperature_2m_max": temp_max[i],
            "temperature_2m_min": temp_min[i],
            "precipitation_sum": precipitation[i],
            "wind_speed_10m_max": wind_speed[i]
        })

    print("\n Step 1: extract task \n")
    print("Columnar data transformation to row dictionaries complete.")
    
    print(f"Extraction step complete: Processed {len(row_records)} daily records for SFO in 2023.")
    logger.info(f"Extraction step completed with {len(row_records)} rows.")
    
    return row_records


# load_raw task
@task(retries=2, retry_delay_seconds=5)
def load_raw(row_records: list):

    logger = get_run_logger()
    logger.info(f"Preparing to load {len(row_records)} raw records into Supabase...")

    if not row_records:
        print("No raw records provided to upsert.")
        return

    # Execute the upsert query using your global 'supabase' variable
    response = (
        supabase.table("weather_raw").upsert(row_records, on_conflict="date").execute()
    )

    # Extract the number of records actually handled by the operation
    upserted_count = len(row_records)

    # Print a confirmation with the upserted row count as requested
    print("\nStep 2: load_raw task\n")
    print(f"Load Raw successful: Upserted {upserted_count} raw rows into weather_raw.")
    logger.info(f"Successfully finished raw storage phase for {upserted_count} entries.")

# transform task
@task(name="transform")
def transform(raw_records: list) -> list:

    logger = get_run_logger()
    print("\nStep 3: transform task\n")
    
    # 1. Fetch dates already in weather_enriched
    existing_response = supabase.table("weather_enriched").select("date", "good_for_running", "confidence", "llm_summary").execute()
    existing_raw_data = existing_response.data
    existing_dates = {row["date"] for row in existing_raw_data}
    
    # Filter out records that already exist
    unprocessed_records = [r for r in raw_records if r["date"] not in existing_dates]
    total_unprocessed = len(unprocessed_records)

   # Explicitly normalize existing historical rows to guarantee a perfectly consistent structure shape
    complete_enrichment_records = []
    for row in existing_raw_data:
        complete_enrichment_records.append({
            "date": str(row["date"]),
            "good_for_running": bool(row["good_for_running"]),
            "confidence": float(row["confidence"]) if row["confidence"] is not None else 1.0,
            "llm_summary": str(row["llm_summary"])
        })

    if total_unprocessed == 0:
        print("All records already processed. Skipping transformation.")
        return complete_enrichment_records
        
    # 2. Load the saved sklearn Pipeline and Metadata files
    model_path = os.path.join("models", "weather_classifier.pkl")
    metadata_path = os.path.join("models", "weather_classifier_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Missing your model files inside the models/ directory!")
        
    ml_pipeline = joblib.load(model_path)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Use the explicit feature names from your metadata JSON file
    feature_list = metadata.get("feature_names")
    if not feature_list or len(feature_list) < 4:
        raise KeyError("Could not find a valid 'feature_names' list inside metadata JSON file.")    
    enrichment_records = []
    
    # 3. Process remaining records with progress printing every 50 records
    for index, record in enumerate(unprocessed_records, start=1):
        
       # Map features safely by exact string name matches instead of index order
        feature_dict = {}
        for feature_name in feature_list:
            # Map input names directly to your JSON's string features safely
            if "temperature_2m_max" in feature_name:
                feature_dict[feature_name] = record["temperature_2m_max"]
            elif "temperature_2m_min" in feature_name:
                feature_dict[feature_name] = record["temperature_2m_min"]
            elif "precipitation_sum" in feature_name:
                feature_dict[feature_name] = record["precipitation_sum"]
            elif "wind_speed_10m_max" in feature_name:
                feature_dict[feature_name] = record["wind_speed_10m_max"]
            else:
                feature_dict[feature_name] = record.get(feature_name, 0.0)

        # Build DataFrame directly from our reliably mapped feature dictionary
        feature_df = pd.DataFrame([feature_dict])
        
        # Run predict and predict_proba on the unprocessed data row
        prediction = int(ml_pipeline.predict(feature_df))
        probabilities = ml_pipeline.predict_proba(feature_df)[0]
        confidence = float(probabilities[prediction])
        
        verdict_str = "Good for running" if prediction == 1 else "Bad for running"
        
        # 4. Call OpenAI API with a graceful error fallback string block
        prompt = (
            f"Weather details: Max Temp {record['temperature_2m_max']}°C, "
            f"Min Temp {record['temperature_2m_min']}°C, Rain {record['precipitation_sum']}mm, "
            f"Wind Speed {record['wind_speed_10m_max']}km/h. "
            f"ML Model Recommendation: {verdict_str}. "
            f"Write a brief one-sentence coaching recommendation explaining why."
        )
        
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an encouraging running coach data assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.7
            )
            llm_summary = completion.choices[0].message.content.strip()

        except Exception as llm_error:
            # Handle LLM errors with a fallback string
            logger.warning(f"LLM API failure encountered on date {record['date']}: {str(llm_error)}")
            llm_summary = f"Weather conditions are predicted to be {verdict_str.lower()} for your run today."
            
        # Structure payload to match database columns
        complete_enrichment_records.append({
            "date": record["date"],
            "good_for_running": bool(prediction),
            "confidence": confidence,
            "llm_summary": llm_summary
        })
        
        # Print progress loop every 50 records as requested
        if index % 50 == 0 or index == total_unprocessed:
            print(f"Transformation Progress Check: Processed {index}/{total_unprocessed} records...")  
   
    print(f"Transform task successfully finished: Processed {total_unprocessed} new enriched records. Total complete set: {len(complete_enrichment_records)}.")

    return  complete_enrichment_records

# load_enriched task
@task(retries=2, retry_delay_seconds=5, name="load_enriched")
def load_enriched(enrichment_records: list):
    # Guards against empty data payloads and idemptotently upserts enriched records 
    # into the weather_enriched Supabase production target table.
    logger = get_run_logger()
    print("\nStep 4: load_enriched task\n")
    
    if not enrichment_records:
        print("No enrichment records to load. Database write skipped.")
        return

    print(f"Connecting to database to push {len(enrichment_records)} predictions into weather_enriched...")
    
    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    upserted_count = len(enrichment_records) 
    
    # Prints a confirmation with the upserted row count
    print(f"Load Enriched successful: Upserted {upserted_count} enrichment records into weather_enriched.")
    logger.info(f"Target cluster synchronization verified for {upserted_count} rows.")


# flow
@flow(name="SFO-Weather-ETL-Pipeline", log_prints=True)
def run_weather_pipeline():
    
    # Orchestrates the sequential tasks of the weather pipeline.
    logger = get_run_logger()
    logger.info("Starting complete Weather ETL Pipeline run...")
    print("Flow started: Executing pipeline sequential graph...")

    # Step 1: Extract data from Open-Meteo
    raw_records = extract()

    # Step 2: Load raw data into Supabase
    load_raw(raw_records)

    # Step 3: transform task
    enriched_records = transform(raw_records)

    # Step 4: Load final predictions into weather_enriched
    load_enriched(enriched_records)

    print("Flow complete: All executed tasks finished successfully.")
    logger.info("Pipeline lifecycle completed.")

if __name__ == "__main__":
    # This block triggers the flow when you run the script directly
    run_weather_pipeline()
