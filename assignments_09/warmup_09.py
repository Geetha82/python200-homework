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
    response = supabase_client.table("weather_raw").upsert(test_record).execute()
    print("Row successfully inserted!")
    return response


if __name__ == "__main__":
    client = get_client()
    insert_test_record(client)

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
