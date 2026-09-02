

import os
import datetime
from datetime import date, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Supabase Connection ---
# Connection Question 1
# =====================================================================
# Q1: What are the two pieces of information supabase-py needs to connect 
#     to your project? Where do you find them in the Supabase dashboard, 
#     and why should they never be hardcoded in a Python script?
#
#A1: 
#     1. The Two Pieces of Information:
#        - Project URL: The unique API URL endpoint for your hosted database instance.
#        - Anon Public Key: The client API key allowing public access to your database.
# 
#     2. Where to find them in the Supabase Dashboard:
#        - Go to Project Settings -> API (found under the gear icon in the sidebar).
#        - Under "Project API keys", copy the 'anon' (public) key.
#        - Under "Project URL", copy the 'URL'.
# 
#     3. Why they should never be hardcoded:
#        - Hardcoding keys compromises security, exposing access if your code is pushed 
#          to public repositories like GitHub. 
#        - Hardcoding limits configuration management, making it difficult to swap 
#          between environments (e.g., development, testing, production) without altering code.

# --- Connection Question 2 ---
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

# --- Connection Question 3 ---
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

# --- CRUD Question 1 ---

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
    
    # Use the supabase-py SDK fluent API to insert the row
    response = (
        supabase.table("weather_raw")
        .insert(test_record)
        .execute()
    )
    
    return response.data

# CRUD Question 1 Comment
# =====================================================================
# Q: What would happen if you ran the function twice? How would you change the call 
#    to make it safe to run multiple times?
# 
# A: 
#     1. What happens if run twice:
#        - The script crashes with a '23505' PostgreSQL exception: duplicate key value 
#          violates unique constraint "weather_raw_pkey". The literal `.insert()` command 
#          will only work if the primary key (date) does not already exist in the table.
# 
#     2. How to make it safe to run multiple times (Idempotency):
#        - Swap the `.insert()` method out for the `.upsert()` method.
#        - Configure it with the parameter `on_conflict="date"`. This tells the database 
#          to overwrite (update) the metrics if the date already exists instead of throwing 
#          an unhandled exception.


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


# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    print("--- Starting Week 9 Warmup Script ---")
    try:
        # 1. Connect securely
        supabase_client = get_client()
        print("Success: Supabase client initialized securely!\n")
        
        # # >>> CHANGED HERE <<<
        # 2. Insert test record calling insert_test_record() directly with try/except to absorb duplicate key crashes safely
        try:
            inserted_data = insert_test_record(supabase_client) # Calling your formal function directly now
            print("Success! Operation response payload:")
            print(inserted_data, "\n")
        except Exception as insert_err:
            print(f"Notice: insert_test_record() skipped/halted as expected (duplicate key row for today already exists): {insert_err}\n")
        
        # 3. Test Range Selection covering the target record (CRUD Q2)
        today = date.today()
        start_date = (today - timedelta(days=1)).isoformat()
        end_date = (today + timedelta(days=1)).isoformat()
        
        print(f"Querying database records from range: {start_date} to {end_date}...")
        records = get_records_by_date_range(supabase_client, start_date, end_date)
        
        print(f"Success! Retrieved {len(records)} record(s):")
        for row in records:
            print(row)
            
    except Exception as err:
        print(f"\nDatabase/Runtime Error: {err}")
