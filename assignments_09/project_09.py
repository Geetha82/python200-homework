# Video link -  https://youtu.be/0oJAR84fEMM


import os
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment configuration from .env file
load_dotenv()

# Initialize cloud database client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing valid connection secrets in your local .env configuration.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Step 1: Extract
def extract_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:

    # Calls the Open-Meteo historical archive API using a clean base_url and 
    # parameter dictionary to safely isolate query strings from path variables.
    print(f"Requesting 2023 timeline from Open-Meteo API (Lat: {lat}, Lon: {lon})...")
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Isolate parameters into a safe native Python dictionary
    query_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "format": "json",
        "timezone": "GMT"
    }
    
    response = requests.get(base_url, params=query_params)
    
    # Catches 400/500 errors early before JSON decoding is attempted
    response.raise_for_status() 
    payload = response.json()
    
    print("\n=== API Response Summary ===")
    print(f"Status Code: {response.status_code}")
    print(f"Coordinates: Lat {payload.get('latitude')}, Lon {payload.get('longitude')}")
    print(f"Elevation: {payload.get('elevation')} meters")
    if "daily" in payload:
        print(f"Extracted Days Count: {len(payload['daily'].get('time', []))}")
    print("============================\n")
    
    return payload

# STEP 2 & 3: TRANSFORM + LOAD
def load_records_to_cloud(raw_api_data: dict):
    
    # Step 2 & 3: Transform + Load
    # Converts the API response from columnar arrays into a list of row dictionaries
    # and loads them idempotently into Supabase database.
    
    daily = raw_api_data.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    wind_speeds = daily.get("wind_speed_10m_max", [])
    
    # --- STEP 2: TRANSFORMATION ---
    records = []
    for i in range(len(dates)):
        records.append({
            "date": dates[i],
            "temperature_2m_max": max_temps[i],
            "temperature_2m_min": min_temps[i],
            "precipitation_sum": precip[i],
            "wind_speed_10m_max": wind_speeds[i]
        })
    
    print("--- Step 2: Transform Verification ---")
    if records:
        print(f"First Record (Index 0):\n{records[0]}")
        print(f"\nLast Record (Index {len(records)-1}):\n{records[-1]}")
    print("--------------------------------------\n")
    
    # --- STEP 3: LOADING (UPSERT) ---
    print(f" Loading {len(records)} records into Supabase 'weather_raw'...")
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Confirmation: Number of rows successfully upserted into the database: {len(response.data)}")   

    # --- AUTOMATED CODE-DRIVEN SECOND RUN & IDEMPOTENCY TEST ---
    print("\n Running Automated Idempotency Verification Test...")
    
    # 1. Fetch exact database count immediately after the initial load
    count_before = supabase.table("weather_raw").select("*", count="exact").execute().count
    print(f"Row count in cloud database after initial load: {count_before}")
    
    # 2. Trigger an immediate second batch upsert with the exact same data payload
    print("Triggering immediate second pipeline run with identical payload...")
    supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    
    # 3. Fetch the final exact database count after the second run completes
    count_after = supabase.table("weather_raw").select("*", count="exact").execute().count
    print(f" Row count in cloud database after second load:  {count_after}")
    
    # 4. Programmatically compare and assert that the row count did not alter or grow
    print("\nCODE-BASED IDEMPOTENCY VALIDATION:")
    if count_before == count_after:
        print(f"SUCCESS: Row counts are IDENTICAL ({count_before} == {count_after}).")
        print("This explicitly proves in code that duplicate entries were prevented.")
    else:
        print(f"ERROR: Row counts drifted! Duplicates were created ({count_before} != {count_after}).")
    print("-" * 70)

# =====================================================================
# STEP 2 REFLECTION: RECORD COUNT DISCREPANCY ANALYSIS
#
# 1. How many records do you expect for a full year, and how many did you get?
#    I expected exactly 365 daily weather records for the full calendar year 
#    of 2023, and the Open-Meteo API successfully extracted and transformed 
#    exactly 365 records (with indices running from 0 to 364).
#
# 2. If the numbers differ, what might explain the discrepancy?
#    While the pipeline extracted exactly 365 records for 2023, the total row 
#    count in the cloud database after loading is 367. This slight discrepancy 
#    of +2 rows is perfectly explained by our development workflow:
#
#    - Warmup Footprint: We previously inserted 2 extra test rows during our 
#      warmup exercise scripts (one literal test record for today's date and 
#      one for tomorrow's date) into the exact same 'weather_raw' table.
#    - Year Variability: If we targeted a leap year (like 2024) instead of 
#      2023, the API extraction itself would yield 366 records due to February 29th.
#    - Network/Data Drops: If an upstream archive weather station went offline 
#      for maintenance, an API might skip days or return null gaps, causing 
#      the final record output to fall short of the expected 365 days.
# =====================================================================

# =====================================================================
# STEP 3 REFLECTION: IDEMPOTENCY CONFIRMATION
#
# What happens when you run the script a second time?
# When running the script a second time, the database successfully returns 
# a confirmation showing that 365 rows were processed, but the total row 
# count in the 'weather_raw' table remains completely unchanged (367 rows).
# No duplicate records are created, and no primary key constraint errors are thrown.
#
# What does this tell you about idempotency?
# This confirms that our pipeline is strictly idempotent. Because we used 
# an `.upsert()` operation with `on_conflict="date"`, the database safely 
# overwrites or skips identical records instead of duplicating rows or crashing. 
# This guarantees that running our data pipeline 100 times will result in 
# the exact same stable database state as running it just once, which is vital 
# for data consistency when restarting failed or interrupted production pipelines.
# =====================================================================

    
# Step 4: Verify
def verify_database_data(supabase_client: Client):
    """
    Runs a series of database queries to check and verify that our weather data 
    was loaded completely and accurately.
    """
    print("\nStep 4: Running Boundary Verification Queries...")
    
    # 1. OPTIMIZED TOTAL ROW COUNT QUERY 
    # Only select 'date' column with count="exact" to save network bandwidth
    count_response = supabase_client.table("weather_raw").select("date", count="exact").execute()
    total_rows = count_response.count if count_response.count is not None else len(count_response.data)
    
    # 2. Earliest and Latest Dates Check
    earliest_res = supabase_client.table("weather_raw").select("date").order("date", desc=False).limit(1).execute()
    latest_res = supabase_client.table("weather_raw").select("date").order("date", desc=True).limit(1).execute()
    
    earliest_date = earliest_res.data[0]["date"] if earliest_res.data else "None"
    latest_date = latest_res.data[0]["date"] if latest_res.data else "None"
    
    # 3. Target Row Lookup (July 4th, 2023 with Nearest Date Fallback)
    target_date = "2023-07-04"
    july_fourth_res = supabase_client.table("weather_raw").select("*").eq("date", target_date).execute()
    
    # STEP 4 TERMINAL OUTPUT SUMMARY
    print("\n================ STEP 4 VERIFICATION RESULTS ================")
    print(f"EXPLICIT TOTAL ROW COUNT QUERY RESULT: {total_rows} rows found.")
    print(f"HISTORICAL TIMELINE BOUNDARIES: Earliest -> {earliest_date} | Latest -> {latest_date}")
    print(f"TARGET RECORD SEARCH FOR {target_date}:")
    
    if july_fourth_res.data:
        print(july_fourth_res.data[0])
    else:
        print(f" Record for {target_date} was missing! Searching for nearest date...")
        closest_before = supabase_client.table("weather_raw").select("*").lt("date", target_date).order("date", desc=True).limit(1).execute()
        closest_after = supabase_client.table("weather_raw").select("*").gt("date", target_date).order("date", desc=False).limit(1).execute()
        
        fallback_records = []
        if closest_before.data:
            fallback_records.append(closest_before.data[0])
        if closest_after.data:
            fallback_records.append(closest_after.data[0])
            
        if fallback_records:
            from datetime import date
            target_parsed = date.fromisoformat(target_date)
            nearest_record = min(
                fallback_records, 
                key=lambda r: abs((date.fromisoformat(r["date"]) - target_parsed).days)
            )
            print(f"Nearest alternative record discovered for date [{nearest_record['date']}]:")
            print(nearest_record)
        else:
            print(" Critical error: No adjacent weather entries exist in the table.")
    print("==============================================================\n")



# MAIN EXECUTION
if __name__ == "__main__":
    TARGET_LAT = 34.0522 
    TARGET_LON = -118.2437
    START_WINDOW = "2023-01-01"
    END_WINDOW = "2023-12-31"
    
    try:
        # Step 1: Extract
        weather_payload = extract_historical_weather(
            lat=TARGET_LAT, 
            lon=TARGET_LON, 
            start_date=START_WINDOW, 
            end_date=END_WINDOW
        )
        
        # Step 2 & 3: Transform + Load (Includes embedded second run comparison check)
        load_records_to_cloud(weather_payload)
        
        # Step 4: Verify
        verify_database_data(supabase)
        
    except Exception as error:
        print(f"Core pipeline execution halted: {error}")
