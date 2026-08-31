
# video link: "https://youtu.be/r2qQh6x_KtI"

import os
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Pipeline Configuration & Connection ---

# This looks up one directory from project_09.py into python200-homework/
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "..", ".env")
load_dotenv(env_path)

def get_supabase_client() -> Client:
    """Initializes the database client using environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing Supabase credentials in .env file.")
    return create_client(url, key)


# --- Step 1: Extract ---

def extract_weather_data() -> dict:

    # Fetches historical daily weather data from the Open-Meteo Archive API for the full year 2023.
    print("Step 1: Extracting 2023 weather data from Open-Meteo API...")
    
    # Base URL for historical weather archive API
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Query parameters matching the fields classification model expects
    params = {
        "latitude": 34.0522,    # Latitude of chosen city (e.g., Los Angeles)
        "longitude": -118.2437, # Longitude of chosen city
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status() # Catch any network or server errors early
    
    payload = response.json()
    daily = payload.get("daily", {})
    
    # Print response summary metrics to terminal
    print("\n--- API Response Summary ---")
    print(f"Status Code: {response.status_code}")
    print(f"Total days extracted: {len(daily.get('time', []))}")
    print("----------------------------\n")
    
    return payload

# --- Step 2: Transform ---

def transform_weather_data(api_payload: dict) -> list[dict]:
   
    # Converts Open-Meteo columnar API response into a list of row dictionaries matching the weather_raw schema columns exactly.
  
    print("Step 2: Transforming API payload into individual row records...")
    
    daily_data = api_payload.get("daily", {})
    total_days = len(daily_data.get("time", []))
    records = []
    
    # Loop through parallel arrays and bundle them into rows
    for i in range(total_days):
        row = {
            "date": daily_data["time"][i],
            "temperature_2m_max": daily_data["temperature_2m_max"][i],
            "temperature_2m_min": daily_data["temperature_2m_min"][i],
            "precipitation_sum": daily_data["precipitation_sum"][i],
            "wind_speed_10m_max": daily_data["wind_speed_10m_max"][i]
        }
        records.append(row)
        
    # Print the first and last records to confirm the conversion layout
    if records:
        print("\nFirst Record (Index 0):")
        print(records[0])
        
        print("\n Last Record (Index -1):")
        print(records[-1])
        print("--------------------------------------------------\n")
        
    return records

# --- Step 2 Comment Questions ---
# Q: How many records do you expect for a full year, and how many did you get? 
#    If the numbers differ, what might explain the discrepancy?

# Answer:
# I expected exactly 365 records because 2023 was a standard non-leap year containing 
# 365 days. The API successfully returned exactly 365 records, matching expectations perfectly.

# If the numbers ever differ, potential explanations include:
# 1. Leap Years: Leap years contain 366 days instead of 365 (e.g., 2024).
# 2. Timezone Offsets: Misconfigured or shifting timezone definitions near boundary dates 
#    can cause data points to be missed or drop off into adjacent days.
# 3. Upstream Missing Data: Weather recording stations occasionally experience local hardware 
#    outages or maintenance windows, leaving data gaps for specific days.


# --- Step 3: Load ---

def load_weather_data(supabase_client: Client, records: list[dict]):

    # Loads all records into Supabase. Uses upsert on conflict of 'date' to keep the database pipeline idempotent.
    print(f" Step 3: Loading {len(records)} records into Supabase 'weather_raw'...")
    
    # load rows in small chunks to avoid server timeouts or payload limits
    batch_size = 100
    total_upserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        
        # .upsert with on_conflict="date" makes this step safe to re-run multiple times
        response = supabase_client.table("weather_raw") \
            .upsert(batch, on_conflict="date") \
            .execute()
            
        total_upserted += len(response.data) if response.data else 0
        
    print(f" Pipeline Complete! Total rows safely upserted: {total_upserted}")


# --- Step 3 Idempotency Reflection ---
# Q: What does running the script a second time and seeing the same row count tell you about idempotency?

# Answer:
# This behavior confirms that our data pipeline is completely idempotent. 
# Because we used the `.upsert(on_conflict="date")` operation, running the script a second 
# time safely updated/overwrote the existing 365 rows with the exact same data instead of 
# creating messy, duplicate rows or crashing with a unique constraint error. 
# It proves that no matter how many times this pipeline accidentally runs or gets 
# restarted, the state of the database will always remain clean, predictable, and correct.

# --- Step 4: Verify --- 
def verify_database_data(supabase_client: Client):
    # Runs a series of database queries to check and verify that our weather data was loaded completely and accurately.
    # Implements a robust fallback search to locate the nearest available date if 2023-07-04 is missing.

    print("\nStep 4: Running Verification Queries...")
    
    # 1. Print the total number of rows in the table
    count_response = supabase_client.table("weather_raw").select("*", count="exact").execute()
    total_rows = count_response.count if count_response.count is not None else len(count_response.data)
    print(f"Total number of rows in 'weather_raw': {total_rows}")
    
    # 2. Print the earliest and latest dates in the table
    earliest_res = supabase_client.table("weather_raw").select("date").order("date", desc=False).limit(1).execute()
    latest_res = supabase_client.table("weather_raw").select("date").order("date", desc=True).limit(1).execute()
    
    earliest_date = earliest_res.data[0]["date"] if earliest_res.data else "None"
    latest_date = latest_res.data[0]["date"] if latest_res.data else "None"
    print(f"Earliest date in table: {earliest_date}")
    print(f"Latest date in table: {latest_date}")
    
    # 3. Print the specific row record for 2023-07-04 (with nearest date fallback)
    target_date = "2023-07-04"
    july_fourth_res = supabase_client.table("weather_raw").select("*").eq("date", target_date).execute()
    
    print(f"\nWeather record for {target_date}:")
    if july_fourth_res.data:
        print(july_fourth_res.data[0])
    else:
        print(f"Record for {target_date} was missing! Searching for the nearest alternative date...")
        
        # NEAREST DATE FALLBACK: Fetch the closest chronological records before and after the target date
        closest_before = supabase_client.table("weather_raw").select("*").lt("date", target_date).order("date", desc=True).limit(1).execute()
        closest_after = supabase_client.table("weather_raw").select("*").gt("date", target_date).order("date", desc=False).limit(1).execute()
        
        fallback_records = []
        if closest_before.data:
            fallback_records.append(closest_before.data[0])
        if closest_after.data:
            fallback_records.append(closest_after.data[0])
            
        if fallback_records:
            # Parse dates using ISO strings to compute absolute mathematical delta distance
            from datetime import date
            target_parsed = date.fromisoformat(target_date)
            
            # Select the record with the minimum absolute day difference from 2023-07-04
            nearest_record = min(
                fallback_records, 
                key=lambda r: abs((date.fromisoformat(r["date"]) - target_parsed).days)
            )
            print(f"Nearest alternative record discovered for date [{nearest_record['date']}]:")
            print(nearest_record)
        else:
            print("Critical: No adjacent weather entries exist in the raw table instance.")
            
    print("--------------------------------------------------\n")



    # --- Run the Complete Pipeline End-to-End ---

if __name__ == "__main__":
    try:
        # 1. Connect to Database
        db_client = get_supabase_client()
        
        # 2. Execute Step 1: Extract
        raw_data = extract_weather_data()
        
        # 3. Execute Step 2: Transform
        prepared_records = transform_weather_data(raw_data)
        
        # 4. Execute Step 3: Load
        load_weather_data(db_client, prepared_records)
        
        # 5. Execute Step 4: Verify
        verify_database_data(db_client)
        
    except Exception as e:
        print(f" Pipeline crashed: {e}")

