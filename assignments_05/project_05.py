import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file so we can read the secret API key
load_dotenv()

# Set up the OpenAI client to talk to the AI model
client = OpenAI()

# ----- Task 1: Setup and System Prompt
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """
    This function sends our list of messages to OpenAI and gets a text reply back.
    We limit the response length using max_completion_tokens to keep things fast.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# MY DESIGN CHOICE (TASK 1 CHECK):
# I made a deliberate choice to use clear numbers (1, 2, 3) and capitalized 
# words like "MUST" for the rules. I did this because AI models follow 
# rules much better when they are listed in a clear order. 
#
# When I read this prompt out loud, it sounds very specific. It tells the 
# AI exactly who to help (career changers) and what it is allowed to talk 
# about (resumes, cover letters, and interviews). It cannot be used as a 
# general calculator or a general coding bot because of these strict rules.

SYSTEM_PROMPT = (
    "You are an expert job application coach. Your role is to help job seekers, "
    "particularly career changers, translate their unique professional experiences "
    "into strong application content.\n\n"
    "CRITICAL BEHAVIORAL CONSTRAINTS:\n"
    "1. You must stay strictly focused on job application materials (such as resumes, "
    "cover letters, and interview preparation).\n"
    "2. At the end of every response where you provide text for the user to use, you "
    "MUST remind the user to review and edit your output before submitting it anywhere.\n"
    "3. You must explicitly acknowledge that you may not know the user's specific "
    "industry norms, and state that the user should use their own judgment."
)

# Start conversation list by giving the AI its role instructions
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# SYSTEM PROMPT CLARITY CHECK (Task 1 Reflection):
# Reading the system prompt aloud confirms it functions as a highly specific 
# briefing rather than a vague assistant. It cannot be used for general tasks 
# like math or creative writing because it explicitly restricts operations to:
#  - Target Audience: Career changers needing skill translation.
#  - Deliverable Scope: Limited exclusively to resumes, cover letters, and interviews.
#  - Explicit Disclaimers: Forcing a review reminder and industry norm acknowledgment.


# ----- Task 2: Bullet Point Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """
    Takes a list of raw resume bullet points, sends them to OpenAI in a structured
    prompt, parses the required JSON response, and outputs them side-by-side.
    """
    # Format the input list into a clean, markdown-delimited string of bullets
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    
    # Structure the prompt with clear behavior expectations and strict JSON constraints
    prompt = f"""
You are a professional resume coach helping a career changer. 
Rewrite each resume bullet point below to be more specific, results-oriented, and compelling. 
Use strong action verbs and imply high-value professional skills. 
Do not invent specific facts or names that aren't implied by the original.

Return ONLY a valid JSON list. Do not wrap the JSON in markdown code blocks like ```json.
Each item in the list must be an object with exactly two keys:
1. "original" - containing the exact original bullet point text.
2. "improved" - containing your rewritten version.

Bullet points to rewrite:
\"\"\"
{bullet_text}
\"\"\"
"""
    # Packages the prompt into the user role format
    messages = [{"role": "user", "content": prompt}]
    
    # Call our Task 1 completion helper
    raw_response = get_completion(messages, temperature=0.8)
    
    try:
        # FIXED: Standalone helper strictly handles JSON string decoding with ZERO internal printing behavior
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return []
        
# TASK 2 REFLECTION & ASSESSMENT COMMENTS:
# 
# 1. Why the starter bullets are weak:
# - They are purely task-focused ("Helped", "Made", "Worked") and lack impact metrics.
# - They do not mention *how* the work was done or *what* organizational benefit resulted.
# - There are no descriptive technical keywords or industry verbs.
# 
# 2. The kinds of changes the model suggested:
# - Swapped passive verbs for strategic ones ("Resolved", "Compiled and presented", "Collaborated").
# - Added high-value professional phrasing ("enhancing satisfaction and retention", "cross-functional team").
# - Framed the tasks around business value ("driving informed decision-making through data analysis", "high-quality deliverables").
# 
# 3. Code Verification Checks:
# - JSON Check: json.loads() succeeded perfectly without errors because we restricted raw text.
# - Layout Check: Both original and improved versions print cleanly with icons side-by-side.
# - Quality Check: The improvements feel meaningfully better and look much more professional.


# ----- Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    """
    Takes a target job title and a summary of the user's background,
    uses few-shot prompting with strong examples to guide the tone,
    and returns a tailored, high-impact cover letter opening paragraph.
    """
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions under pressure using incomplete information — which turns out to be excellent training for data analysis. I recently completed a data analytics program where I built dashboards tracking patient outcomes across departments. I'm excited to bring that combination of clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions get made by people who had never processed a wire transfer or resolved a failed ACH batch. That frustration turned into curiosity, and two years of self-teaching Python later, I'm ready to be on the other side of those decisions. I'm applying to [Company] because your work on payment infrastructure is exactly where my domain expertise and new technical skills intersect.

Now write an opening paragraph for this person. Remember, you MUST strictly limit your output to a single paragraph containing exactly 3 to 5 sentences max:
Role: {job_title}
Background: {background}
Opening:
"""
    # Wrap the few-shot prompt string into a user message
    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)

# TASK 3 REFLECTION & ASSESSMENT COMMENTS:
# 
# 1. Why I chose these particular examples:
#    - Both examples explicitly feature professionals transitioning from 
#      traditional industries (Nursing, Retail Banking) into tech roles.
#    - They show how to proudly pivot past experiences into modern technical 
#      advantages rather than treating a non-traditional background as a flaw.
# 
# 2. What the few-shot pattern helps control in the output:
#    - It strictly dictates the structure, length (3-5 sentences), and style.
#    - It eliminates tired, automated clichés like "To Whom It May Concern, I am 
#      writing to express my enthusiastic interest in your open vacancy..."
#    - It forces the AI to use concrete project/tool references (like Prefect and Pandas) 
#      instead of hiding behind vague adjectives.
#
# 3. Code Verification Checks:
#    - Tailored Check: The paragraph actively blends teaching skills ("breaking down 
#      complex concepts") with data tools, keeping it personal.
#    - Integrity Check: The model strictly used the provided tools (Prefect and Pandas) 
#      and did not make up an arbitrary university degree or fake previous employer.

# ----- Task 4: Moderation Check
def is_safe(text: str) -> bool:
    """
    Sends the user's text to OpenAI's safety system.
    Returns True if the text is safe, and False if it breaks safety rules.
    """
    # Ask the moderation API to look at our text
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    
    # Check if the very first result item was flagged as unsafe
    flagged = result.results[0].flagged
    
    # If the text is dangerous or inappropriate, warn the user and return False
    if flagged:
        print("\n[Safety Warning]: I can only assist with professional career development and job applications.")
        print("Please rephrase your request to keep it professional and workplace-appropriate.")
        return False
        
    # If everything is perfectly safe, return True
    return True

# TASK 5: THE CHATBOT LOOP (UNIFIED CONVERSATIONAL STATE MACHINE)
def run_chatbot():

    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
        
    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print(" 1. Rewriting resume bullet points")
    print(" 2. Drafting a cover letter opening")
    print(" 3. Any other questions about your application")
    print("\nType 'quit' or 'exit' at any time to exit.\n")
    
    while True:
        # Every single input uses this unified line to maintain conversational continuity
        user_input = input("You: ").strip()
        
        # 2. Handle exit condition instantly
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break
            
        # 3. Skip blank lines safely
        if not user_input:
            continue
            
        # 4. Run safety filtration block before processing states
        if not is_safe(user_input):
            continue  
            
        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            # Append user turn to history to maintain clean conversation memory consistency
            messages.append({"role": "user", "content": user_input})
            
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            
            # Bullet points are gathered locally via a clean, nested inner loop
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
                    
            if raw_bullets:
                print("\nJob Application Helper: Upgrading your bullet points now...")
                
                # Call rewrite_bullets() helper (which calculates and parses natively)
                results = rewrite_bullets(raw_bullets)
                
                # Handles ALL side-by-side display behavior cleanly here in one single place
                if results:
                    print("\n=== RESUME BULLET POINT UPGRADES ===")
                    for i, item in enumerate(results, start=1):
                        print(f"\n[Pair #{i}]")
                        print(f" Original: {item.get('original')}")
                        print(f" Improved: {item.get('improved')}")
                        print("  " + "-" * 40)
                    print("====================================")
                    
                    # Store exact output text back to context memory array safely
                    assistant_record = "I optimized your resume bullet points. Upgrades:\n" + "\n".join(
                        f"Original: {item.get('original')} -> Improved: {item.get('improved')}" for item in results
                    )
                    messages.append({"role": "assistant", "content": assistant_record})
                else:
                    print("\nJob Application Helper: Could not parse upgrades.")
                    messages.append({"role": "assistant", "content": "Failed to rewrite bullet points."})
            else:
                print("\nJob Application Helper: No bullet points received.")
                messages.append({"role": "assistant", "content": "No bullets provided."})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            messages.append({"role": "user", "content": user_input})
            
            # FIXED: Reverted to linear, standard nested input prompts with zero state handling
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            
            if job_title and background:
                print("\nJob Application Helper: Crafting your high-impact opening paragraph...")
                
                # Loop directly calls the task function and prints the paragraph result cleanly
                cover_letter_result = generate_cover_letter(job_title, background)
                
                print("\n=== GENERATED COVER LETTER OPENING ===")
                print(cover_letter_result)
                print("========================================")
                
                messages.append({"role": "assistant", "content": cover_letter_result})
            else:
                print("\nJob Application Helper: Both job title and background are required.")
                messages.append({"role": "assistant", "content": "Canceled cover letter due to blank field data."})
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            
            # - Call get_completion(messages)
            reply = get_completion(messages, temperature=0.7)
            
            # - Print the reply
            print(f"\nJob Application Helper: {reply}\n")
            
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": reply})

        
# =====================================================================
# MAIN APPLICATION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    run_chatbot()

# TASK 6: ETHICS REFLECTION
# Format Chosen: Option A — Comment block

# 1. How might training data produce biased advice or favor certain backgrounds?
# Because the underlying AI model was trained primarily on a large corpus of text 
# from the internet, it heavily favors standard corporate Western communication 
# styles, white-collar industry jargon, and formal American English. This can 
# produce biased advice by penalizing applicants from cultures that value modesty, 
# as the model naturally pushes for aggressive self-promotion and assertive verbs. 
# It may also offer weaker, more generic advice for local non-traditional fields, 
# blue-collar jobs, or specialized global industries that are underrepresented 
# in standard online corporate literature.
#
# 2. What could go wrong if a job-seeker submitted unreviewed output to an employer?
# Submitting unreviewed output poses a massive risk of "hallucinations," where the 
# model invents fake metrics, names placeholder tools, or fabricates job responsibilities 
# to make a bullet point sound more compelling. An unreviewed document might also 
# contain literal placeholders like "[Company Name]" or "[Insert Year]," which looks 
# incredibly unprofessional. If an employer notices these errors or interviews the 
# candidate on an AI-invented technical credential they do not actually possess, 
# the candidate will instantly lose credibility and be disqualified for dishonesty.
#
# 3. What is one guardrail you would add if you were deploying this professionally?
# If deploying this tool professionally, I would add a user interface (UI) guardrail 
# that prevents copy-pasting until an explicit verification checkbox is clicked. 
# Specifically, the web interface would display a popup modal when the user clicks 
# "Copy Text." This modal would show a warning stating: "AI text can contain inaccurate 
# metrics or hallucinated skills. Please verify that every statement is 100% accurate 
# to your personal history." The user would have to check a box reading, "I have reviewed 
# this text for truthfulness," before the system unlocks the clipboard copy function.
# =====================================================================



