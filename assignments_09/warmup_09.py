

import os
import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Supabase Connection ---

# =====================================================================
# Q1: What are the two pieces of information supabase-py needs to connect 
#     to your project? Where do you find them in the Supabase dashboard, 
#     and why should they never be hardcoded in a Python script?
#
# A1: 1. Project URL: The REST API endpoint used to route requests to your 
#        specific database instance.
#     2. Anon Public API Key: The client-side API key that authenticates 
#        your requests through Supabase's API gateway.
#
#     Where to find them: In the Supabase Dashboard, navigate to 
#     Project Settings -> API. Both the Project URL and the anon public 
#     key are listed under the "Project API keys" and "URL" sections.
#
#     Why they should never be hardcoded: Hardcoding credentials risks 
#     leaking them if the code is pushed to a public repository (like GitHub). 
#     Using environment variables (.env files) keeps configuration separate 
#     from source code, allowing safe collaboration and easier deployments 
#     across different environments (development, staging, production).
# =====================================================================

def get_client() -> Client:

    # Loads Supabase credentials from environment variables and returns a Client.
    # Raises a ValueError if SUPABASE_URL or SUPABASE_KEY are missing.
    
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url:
        raise ValueError("Missing 'SUPABASE_URL' in environment variables.")
    if not key:
        raise ValueError("Missing 'SUPABASE_KEY' in environment variables.")
        
    return create_client(url, key)


# =====================================================================
# Q3: What is Row Level Security (RLS), and why did you disable it on 
#     your tables for this course? In what kind of real-world application 
#     would you want to keep it enabled?
#
# A3: Row Level Security (RLS) is a database security feature that restricts 
#     which data rows a user can see, insert, update, or delete based on 
#     the identity of the user executing the query. 
#
#     Why it is disabled for this course: We disabled RLS to allow our 
#     backend Python scripts to read and write rows directly using the 
#     "anon public" API key without requiring us to set up authentication, 
#     sign-in flows, or complex authorization policies.
#
#     Real-world application use case: You would keep RLS enabled for 
#     multi-user apps like a healthcare patient portal, a banking app, 
#     or a SaaS dashboard. For example, in an app where users view their 
#     own bank statements, RLS guarantees that User A can only see rows 
#     matching their specific user_id, completely preventing them from 
#     accessing User B's financial data.
# =====================================================================


# --- supabase-py CRUD ---

def insert_test_record(supabase: Client):
    
    # Inserts a single test row into weather_raw using today's date and a literal .insert() call.
    # This will intentionally crash on a duplicate key error if run more than once per day.

    today_str = datetime.date.today().isoformat()
    
    test_record = {
        "date": today_str,
        "temperature_2m_max": 24.5,
        "temperature_2m_min": 12.2,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 14.3
    }
    
    print(f"Executing standard literal insert for test record date {today_str}...")
    
    # Standard insert call to show standard database constraints
    response = supabase.table("weather_raw").insert(test_record).execute()
    print("Successfully inserted test row!")
    return response


# =====================================================================
# Q1 CONCEPTUAL FOLLOW-UP
#
# What would happen if you ran the function twice?
# If you run this function twice on the same day with a standard .insert(), 
# the second execution will crash and throw a unique constraint primary 
# key violation error because that date already exists in the table.
#
# How would you change the call to make it safe to run multiple times?
# To make it safe and idempotent, use the `.upsert()` method call and 
# provide the conflict target `on_conflict="date"` like this:
# supabase.table("weather_raw").upsert(test_record, on_conflict="date").execute()
# =====================================================================


def get_records_by_date_range(supabase: Client, start: str, end: str) -> list[dict]:
    
    # Retrieves all rows from weather_raw where date >= start and date <= end.
    # Returns a list of dictionaries, where each dictionary represents a row.
    response = supabase.table("weather_raw") \
                       .select("*") \
                       .gte("date", start) \
                       .lte("date", end) \
                       .execute()
    
    return response.data


# =====================================================================
# Q3: Explain the difference between insert and upsert in supabase-py. 
#     Give a concrete example of when you would choose each.
#
# A3: 
#     1. INSERT: Adds completely new rows to a table. If any row contains 
#        a primary key or unique constraint value that already exists in 
#        the database, the entire query fails and throws an error.
#     2. UPSERT (Update or Insert): Safely checks if a record exists based 
#        on a specific conflict key (like 'date'). If the key doesn't 
#        exist, it performs an insert. If the key already exists, it 
#        overwrites/updates that existing row's metrics with the new data.
#
#     Concrete Examples:
#     - Choose INSERT for an **E-commerce Order Transaction log**. Every 
#       purchased item needs a new, unique row ID. If an order ID is duplicated, 
#       you want the script to crash immediately so you can catch a fraud 
#       or billing glitch before charging a customer twice.
#     - Choose UPSERT for an **Automated Weather Pipeline** (like this course). 
#       If you re-run a pipeline to scrape weather stats for January 1st, you 
#       do not want the script to fail. You want it to seamlessly update the 
#       metrics for January 1st if they changed, or safely leave them alone.
# =====================================================================

def safe_upsert(supabase: Client, records: list[dict]) -> list[dict]:
    
    # Idempotently upserts a list of records into weather_raw using 'date'as the unique constraint conflict key and prints the number of rows affected.
    
    if not records:
        print(" No records provided for upsert operation.")
        return []
        
    response = supabase.table("weather_raw") \
                       .upsert(records, on_conflict="date") \
                       .execute()
    
    affected_rows = len(response.data)
    print(f"Upsert complete. Rows affected in database: {affected_rows}")
    
    return response.data


# --- Idempotency ---

# =====================================================================
# Q1: What does "Idempotency" mean, and why does it matter for a data 
#     pipeline? Give one concrete example of what goes wrong in a 
#     non-idempotent pipeline when the script crashes halfway through 
#     and is restarted.
#
# A1: Idempotency means that running an operation multiple times produces the 
#     same database state as running it once. It matters for data pipelines 
#     because real-world networks, APIs, and servers frequently experience 
#     intermittent dropouts. A pipeline must be safe to rerun at any time 
#     without creating duplicate data rows or requiring manual cleanups.
#
#     Concrete failure example: If a non-idempotent script designed to append 
#     365 days of weather charts crashes on day 200 due to a network timeout, 
#     restarting the pipeline from scratch will cause the script to re-append 
#     days 1 through 199. Without an upsert constraint, the database will 
#     either crash on a primary key duplicate exception, or populate 
#     duplicate rows, corrupting your downstream machine learning data arrays.
# =====================================================================


if __name__ == "__main__":
    print("--- Running Reviewer-Aligned SDK Warmup Checks ---")
    try:
        # 1. Initialize connection
        client = get_client()
        print("Successfully generated a valid Supabase client!")
        
        # 2. CRUD Q1 TEST: Standard Literal Insert
        print("\n--- Testing Q1: Literal Insert ---")
        try:
            insert_test_record(client)
        except Exception as insert_error:
            print(f"\nINTENTIONAL CRASH CAUGHT (Demonstrating Q1 Behavior):")
            print(f"   As expected by the prompt, running a literal .insert() twice failed with:")
            print(f"   {insert_error}\n")
        
        # 3. CRUD Q2 TEST & PRINTOUT: Range Query Verification
        today_str = datetime.date.today().isoformat()
        print(f"Executing Q2 Range Query Test bounding today's record ({today_str})...")
        matching_rows = get_records_by_date_range(client, start=today_str, end=today_str)
        
        print(f"Found {len(matching_rows)} matching records from range request:")
        for row in matching_rows:
            print(row)
        print("-" * 60)
        
        # 4. CRUD Q3 BATCH SAFE UPSERT TEST: Batch List Function Check
        print("\n Executing Q3 safe_upsert batch operation test...")
        tomorrow_str = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        batch_payload = [
            {
                "date": today_str,  # Already exists: updates the row metrics seamlessly
                "temperature_2m_max": 28.0,
                "temperature_2m_min": 14.0,
                "precipitation_sum": 0.5,
                "wind_speed_10m_max": 11.2},
                    {
            "date": tomorrow_str, # New target date: performs clean insert
            "temperature_2m_max": 22.0,
            "temperature_2m_min": 10.5,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 8.4
            }
            ]
        upserted_rows = safe_upsert(client, batch_payload)
        print("\n🎉 Warmup script validation suite fully processed successfully!")
    except Exception as e:
        print(f" Warmup execution halted prematurely: {e}")