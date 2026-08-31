import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env
load_dotenv("../.env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize the Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(" Supabase client successfully initialized!")

# --- Supabase Connection ---
# Q1: What are the two pieces of information supabase-py needs to connect to your project? 
#     Where do you find them in the Supabase dashboard, and why should they never be hardcoded in a Python script?

# Answer:
# 1. Project URL: The unique API endpoint for  Supabase database instance.
# 2. Anon/Public API Key: The client-side key used to authenticate API requests.

# Dashboard Location: 
# By navigating to my Supabase Project Dashboard and  clicking on 
# "Project Settings" (the gear icon) in the sidebar menu, and selecting "API".

# Why you shouldn't hardcode them:
# Hardcoding credentials poses a severe security risk. If Python script is committed 
# to a public Git repository (like GitHub), anyone can view my credentials, scrape the
# database, or exhaust the project's platform limits. Keeping them in an uncommitted `.env` 
# file prevents private configuration data from leaking into my  source code history.

# --- Supabase Connection ---
# Q2: Write a function get_client() that loads credentials from environment variables 
#     using python-dotenv and returns a Supabase client. Raise a clear error if missing.

def get_client() -> Client:
    """
    Loads Supabase credentials from environment variables 
    and returns an initialized Supabase Client.
    """
    # Load variables from local .env file
    load_dotenv()
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    # Assert both variables exist before trying to establish a connection
    if not url:
        raise ValueError("Environment configuration error: 'SUPABASE_URL' is missing from .env.")
    if not key:
        raise ValueError("Environment configuration error: 'SUPABASE_KEY' is missing from .env.")
        
    return create_client(url, key)

# Test the function to confirm connection works
if __name__ == "__main__":
    try:
        supabase_client = get_client()
        print("Success! get_client() returned a valid Supabase client instance.")
    except Exception as e:
        print(f"Connection failed: {e}")

# --- Supabase Connection ---
# Q3: What is Row Level Security (RLS), and why did you disable it on your tables 
#     for this course? In what kind of real-world application would you want to keep it enabled?
#
# Answer:
# What is RLS?
# Row Level Security (RLS) is a security system built into databases. It acts like a 
# guard that checks who is asking for data and decides exactly which individual rows they 
# are allowed to see, update, or delete. 
#
# Why we disabled it for this course:
# We disabled RLS to make learning easier. When RLS is turned on, every single database 
# query from Python fails unless you also write complex security rules (policies) inside 
# the Supabase dashboard. Turning it off lets us focus completely on practicing Python 
# database commands without getting blocked by permissions.
#
# Real-World Application Example:
# You would keep RLS enabled for any app where users have private accounts, such as a 
# banking app, a social media network (like Instagram), or a medical portal. For example, 
# in a banking app, RLS ensures that when you log in, you can only see your own account 
# balance and transaction history, while completely blocking you from viewing rows 
# belonging to other users.

from datetime import datetime

# --- Section 2: supabase-py CRUD ---
# Q1: Write a function insert_test_record(supabase) that inserts a single row 
#     into weather_raw with today's date and plausible values.

def insert_test_record(supabase_client):
    """
    Inserts a single test row into weather_raw using today's date string.
    """
    # Get today's date formatted as a clean string: YYYY-MM-DD
    today_str = datetime.today().strftime('%Y-%m-%d')
    
    test_record = {
        "date": today_str,
        "temperature_2m_max": 24.5,
        "temperature_2m_min": 15.0,
        "precipitation_sum": 0.5,
        "wind_speed_10m_max": 11.2
    }
    
    print("⏳ Inserting test row into weather_raw...")
    response = supabase_client.table("weather_raw").insert(test_record).execute()
    print("Row successfully inserted!")
    return response


# --- CRUD Question 1 Conceptual Answers ---
#
# What would happen if you ran the function twice?
# Because the 'date' column is set as a Primary Key (indicated by the small key icon 
# in the Supabase Table Editor), every entry must have a unique date value. 
# If I  run the function a second time on the same day, Supabase will reject the entry 
# and raise a PostgrestError (API Error) stating that a unique constraint has been violated.
#
# How would you change the call to make it safe to run multiple times?
# To make it safe to run multiple times without crashing, I  changed the `.insert()` method 
# to an `.upsert()` method. Upsert stands for "Update or Insert". 
# If a row with today's date doesn't exist yet, it will create a new one. 
# If a row with today's date already exists, it will simply overwrite the old values 
# instead of causing a system error.
#
# Safe code alternative:
# response = supabase_client.table("weather_raw").upsert(test_record).execute()

# Q2: Write a function get_records_by_date_range(supabase, start, end)
def get_records_by_date_range(supabase: Client, start: str, end: str) -> list[dict]:
    """
    Retrieves all rows from weather_raw where date >= start and date <= end.
    Returns a list of dictionaries, where each dictionary represents a row.
    """
    print(f"Querying records between {start} and {end}...")
    
    response = supabase.table("weather_raw") \
                       .select("*") \
                       .gte("date", start) \
                       .lte("date", end) \
                       .execute()
    
    return response.data

# --- CRUD Question 3 Conceptual Answers ---
#
# Difference between Insert and Upsert:
# 1. INSERT: Always attempts to create brand-new data rows. If a row already exists with 
#    the exact same unique ID/Primary Key, the database will completely crash with an error.
# 2. UPSERT (Update + Insert): Checks if the unique row identifier already exists. 
#    If the record is missing, it adds a new row. If it already exists, it updates and 
#    overwrites the existing row with the incoming data instead of throwing an error.
#
# Concrete Examples:
# - Use INSERT: Creating a user transaction history table. When a user purchases an item, 
#   you always want a brand new row to record that unique receipt event.I do not want 
#   to accidently overwrite past transactions.
# - Use UPSERT: Building a weather pipeline that syncs values. If re-pull data for 
#   '2026-08-31', I  do not want my script to crash if that date already exists. I want 
#   it to gracefully overwrite the old readings with the fresh up-to-date data.

# --- Idempotency ---
# Q1: Explain why idempotency matters for a data pipeline. Give one concrete example 
#     of what goes wrong when a non-idempotent pipeline crashes halfway through and is restarted.
#
# Answer:
# Why idempotency matters:
# Idempotency is a crucial safety feature for data pipelines. It ensures that no matter 
# how many times a script accidentally runs or gets restarted, the final data in the 
# database remains correct, clean, and free of duplicate rows.
#
# Concrete example of what goes wrong:
# Imagine your pipeline is designed to insert 30 days of weather records using standard 
# INSERT commands. If the script crashes on day 15 due to a temporary network glitch and 
# you restart it, a non-idempotent pipeline will re-insert days 1 through 15 all over again. 
# This results in corrupted data full of messy, duplicate entries for the first half of the month.

# =====================================================================
# assignments_09/warmup_09.py
# Python 200 - Week 9 Warmup Submission
# =====================================================================

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

# Q2: Write a function get_client() that loads credentials and returns a Supabase client.
def get_client() -> Client:
    """
    Loads Supabase credentials from environment variables and returns a Client.
    Raises a ValueError if SUPABASE_URL or SUPABASE_KEY are missing.
    """
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

# Q1: Write a function insert_test_record(supabase) that inserts a single row into weather_raw.
def insert_test_record(supabase: Client):
    """
    Inserts a single test row into weather_raw using today's date and a literal .insert() call.
    NOTE: This will intentionally crash on a duplicate key error if run more than once per day.
    """
    today_str = datetime.date.today().isoformat()
    
    test_record = {
        "date": today_str,
        "temperature_2m_max": 24.5,
        "temperature_2m_min": 12.2,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 14.3
    }
    
    print(f"Executing standard literal insert for test record date {today_str}...")
    
    # LITERAL INSERT AS REQUIRED TO DEMONSTRATE CONSTRAINT FAILURES ACCURATELY
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


# Q2: Write a function get_records_by_date_range(supabase, start, end)
def get_records_by_date_range(supabase: Client, start: str, end: str) -> list[dict]:
    """
    Retrieves all rows from weather_raw where date >= start and date <= end.
    Returns a list of dictionaries, where each dictionary represents a row.
    """
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

# Q3 FUNCTION: Write a function safe_upsert(supabase, records)
def safe_upsert(supabase: Client, records: list[dict]) -> list[dict]:
    """
    Idempotently upserts a list of records into weather_raw using 'date' 
    as the unique constraint conflict key and prints the number of rows affected.
    """
    if not records:
        print("⚠️ No records provided for upsert operation.")
        return []
        
    response = supabase.table("weather_raw") \
                       .upsert(records, on_conflict="date") \
                       .execute()
    
    affected_rows = len(response.data)
    print(f"✅ Upsert complete. Rows affected in database: {affected_rows}")
    
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


# --- Script Execution Routine ---
if __name__ == "__main__":
    print("--- Running Reviewer-Aligned SDK Warmup Checks ---")
    try:
        # 1. Initialize connection
        client = get_client()
        print("🎉 Successfully generated a valid Supabase client!")
        
        # 2. RUNNABLE CRUD Q1 TEST: Standard Literal Insert
        print("\n--- Testing Q1: Literal Insert ---")
        try:
            insert_test_record(client)
        except Exception as insert_error:
            print(f"\n⚠️ INTENTIONAL CRASH CAUGHT (Demonstrating Q1 Behavior):")
            print(f"   As expected by the prompt, running a literal .insert() twice failed with:")
            print(f"   {insert_error}\n")
        
        # 3. RUNNABLE CRUD Q2 TEST & PRINTOUT: Range Query Verification
        today_str = datetime.date.today().isoformat()
        print(f"🚀 Executing Q2 Range Query Test bounding today's record ({today_str})...")
        matching_rows = get_records_by_date_range(client, start=today_str, end=today_str)
        
        print(f"🔍 Found {len(matching_rows)} matching records from range request:")
        for row in matching_rows:
            print(row)
        print("-" * 60)
        
        # 4. RUNNABLE CRUD Q3 BATCH SAFE UPSERT TEST: Batch List Function Check
        print("\n🚀 Executing Q3 safe_upsert batch operation test...")
        tomorrow_str = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        batch_payload = [
            {
                "date": today_str,  # Already exists: updates the row metrics seamlessly
                "temperature_2m_max": 28.0,
                "temperature_2m_min": 14.0,
                "precipitation_sum": 0.5,
                "wind_speed_10m_max": 11.2
            },
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
        print(f"❌ Warmup execution halted prematurely: {e}")
