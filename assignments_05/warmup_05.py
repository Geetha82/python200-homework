import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ----- SYSTEM CONFIGURATION & INITIALIZATION ------
# Load the environment variables .env file
load_dotenv()

# Instantiate the OpenAI client
client = OpenAI()
print("[System Log]: Warmup Environment Successfully Loaded.\n")


# --- The Chat Completions API ---

# API Q1
print("\n [API Q1]: Testing Stateless Chat Completion Request")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
# Extract precise data fields using correct index [0]
text_response = response.choices[0].message.content.strip()
model_used = response.model
tokens_count = response.usage.total_tokens

# Print each output with explicit labels as requested
print(f"[API Q1 Response Text]:\n{text_response}\n")
print(f"[API Q1 Model Responded]: {model_used}")
print(f"[API Q1 Total Tokens Used]: {tokens_count}\n")

# API Q2
print("\n[API Q2]: Testing Diverse Temperature Array Configurations")

prompt_q2 = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response_q2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_q2}],
        temperature=temp
    )
    
    # Extract the response text safely
    text_output = response_q2.choices[0].message.content.strip()
    
    # Print each output clearly labeled with its specific temperature setting
    print(f"Temperature [{temp}]: {text_output}")
    print("-" * 30)

# --- API Q2 ---
print("\n[API Q2]: Testing Diverse Temperature Array Configurations")
prompt_q2 = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response_q2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_q2}],
        temperature=temp
    )
    text_output = response_q2.choices[0].message.content.strip()
    print(f"Temperature [{temp}]: {text_output}")
    print("-" * 30)

# --- API Q2 Reflection Commentary ---
# - At temperature 0, the output is completely direct and deterministic, returning 
#   just a single business name in quotes with no conversational introductory text.
# - At temperature 0.7, the model introduces creative variance by switching to a structured, 
#   numbered list of 10 diverse naming ideas wrapped in friendly conversational padding.
# - At temperature 1.5, the high randomness completely changes the response structure again; 
#   instead of listing options, it isolates a single name and adds an unprompted, explanatory 
#   paragraph breaking down the strategic business meaning and reasoning behind the choice.
#
# Which temperature would you use if you needed a consistent, reproducible output?
# I would use temperature 0. Setting the temperature to 0 removes token prediction randomness 
# and forces the model to always select the mathematically highest-probability words. This 
# guarantees that running the exact same prompt multiple times will yield identical results.


# API Q3
print("\n [API Q3]: Requesting Multiple Completions (n=3) in One API Call")

response_q3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

# Iterate over response.choices and print each one with a clear index label
for index, choice in enumerate(response_q3.choices):
    fact_text = choice.message.content.strip()
    print(f"Completion #{index + 1}: {fact_text}")
    print("-" * 30)


# API Q4
print("\n [API Q4]: Testing Token Ceiling Guardrails via max_tokens=15")

response_q4 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

# Extract content and finish reason metadata
truncated_text = response_q4.choices[0].message.content.strip()
finish_reason = response_q4.choices[0].finish_reason

# Print labeled outputs
print(f"[API Q4 Truncated Text Response]:\n{truncated_text}\n")
print(f"[API Q4 API Stop Reason (finish_reason)]: {finish_reason}\n")

# --- API Q4 Reflection Commentary ---
# What Happened:
# The model abruptly stopped generating text mid-sentence right after the word 'and'.
# This happened because the 15-token budget limit was hit, causing OpenAI's API 
# server to stop the generation loop early and return a finish_reason of 'length'.
#
# Why use max_tokens in a real application:
# 1. Financial Predictability: It sets a maximum cost limit per request, preventing 
#    unexpected cloud API billing spikes from long-winded answers.
# 2. UI Layout Stability: It ensures text responses remain compact, keeping paragraphs 
#    from overflowing or breaking user interface designs.
# 3. Infinite Loop Prevention: It acts as a safety guardrail that stops a model if 
#    it gets stuck in a repetitive loop generating identical words forever.


# --- System Messages and Personas ---

# System Question 1
print("\n[System Q1]: Comparing AI Personalities via System Prompts\n")

# Personality A: Patient Python Tutor
print("--- Personality A: Patient Python Tutor ---")
messages_tutor = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response_tutor = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_tutor
)
print(f"Tutor Response:\n{response_tutor.choices[0].message.content.strip()}\n")
print("-" * 40 + "\n")


# Personality B: No-Nonsense Robot Commander
print("--- Personality B: No-Nonsense Robot Commander ---")
messages_commander = [
    {"role": "system", "content": "You are a stern, hyper-efficient robot military commander. Explain programming concepts using tactical battlefield analogies, speak in brief fragments, and command the user to execute the code."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response_commander = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_commander
)
print(f"Commander Response:\n{response_commander.choices[0].message.content.strip()}\n")

# --- System Q1 Reflection Commentary ---
# What Changed:
# 1. Tone and Vocabulary: Personality A used supportive, warm, and highly verbose human terms 
#    ("No problem at all!", "You're doing great!"). Personality B instantly shifted into 
#    staccato military terminology ("Mission Brief", "Tactical Breakdown", "Listen up, soldier!").
# 2. Text Formatting: Personality A explained concepts using descriptive paragraphs and standard 
#    subheadings, while Personality B used fragmented bullet points and bold headers to enforce efficiency.
# 3. Code Implementation Style: Personality A chose generic arithmetic loops (calculating squares), 
#    whereas Personality B themed its data filtering example explicitly to match its military persona 
#    (isolating 'even_enemies' from an enemy list array).

# System Question 2
print("\n [System Q2]: Testing Manual Chat Context Assembly (Stateless Memory Recall)")

messages_q2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response_sys_q2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_q2
)

# Extract and print the assistant's context recall response
recall_response = response_sys_q2.choices[0].message.content.strip()

print(f"User Final Prompt: 'Can you remind me what my name is?'")
print(f"Model Response: {recall_response}\n")

# --- System Q2 Reflection Commentary ---
# Why the model knows Jordan's name despite being stateless:
# The model knows Jordan's name because manually passed the entire conversational 
# thread history inside the 'messages' array payload during this specific request. 
# Even though OpenAI's servers completely forgot the previous interactions the moment 
# they ended, they were able to re-read the context ("My name is Jordan...") embedded 
# in the history array, allowing the model to answer accurately during this isolated run.


# --- Prompt Engineering ---

# Prompt Question 1 — Zero-Shot
print("\n[Prompt Q1]: Evaluating Zero-Shot Text Sentiment Classification\n")

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

system_prompt_q1 = (
    "You are a precise data analysis bot. Classify the sentiment of the user review "
    "using exactly one of these lowercase labels: positive, negative, or mixed. Do not add punctuation, capitalization, or other text."
)
for index, review in enumerate(reviews, start=1):
    response_prompt_q1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt_q1},
            {"role": "user", "content": f"Review: {review}"}
        ],
        temperature=0.0  # Zero randomness for clean analytical output
    )
    
    classification = response_prompt_q1.choices[0].message.content.strip()
    
    print(f"Review #{index}: '{review}'")
    print(f"Zero-Shot Label Classification: {classification}")


# Prompt Question 2 — One-Shot
print("\n[Prompt Q2]: Evaluating One-Shot Text Sentiment Classification\n")

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

system_prompt_q2 = (
    "You are a precise data analysis bot. Classify the sentiment of the user review "
    "using exactly one of these lowercase labels: positive, negative, or mixed. Follow the exact "
    "formatting template and casing shown in the example."
)

for index, review in enumerate(reviews, start=1):
    # Constructing a structured prompt string housing exactly one format training example
    one_shot_prompt = f"""
    Example:
    Review: "Fast shipping but the item arrived damaged."
    Sentiment: mixed

    Review: "{review}"
    Sentiment:"""

    response_prompt_q2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt_q2},
            {"role": "user", "content": one_shot_prompt}
        ],
        temperature=0.0  # Lock down determinism to stick strictly to the formatting example
    )
    
    classification = response_prompt_q2.choices[0].message.content.strip()
    
    print(f"Review #{index}: '{review}'")
    print(f"One-Shot Label Classification: {classification}")
    print("-" * 40)

# --- Prompt Q2 Reflection Commentary ---
# Observation:
# Yes, adding the one-shot example directly changed the format of the output. 
# In Q1 (Zero-Shot), the model relied entirely on its internal defaults and the system 
# instructions, resulting in strict uppercase outputs (e.g., 'POSITIVE', 'MIXED'). 
# In Q2 (One-Shot), the model mimicked the exact casing of the provided example 
# ('Sentiment: mixed'), causing it to switch to lowercase outputs ('positive', 'mixed') 
# for matching cases, though it inconsistently left Review #2 as 'NEGATIVE'. 
# This shows how strongly a model mirrors the style and case formatting of examples.


# Prompt Question 3 — Few-Shot
print("\n[Prompt Q3]: Evaluating Few-Shot Text Sentiment Classification\n")

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

system_prompt_q3 = (
    "You are a precise data analysis bot. Classify the sentiment of the user review "
    "using exactly one of these lowercase labels: positive, negative, or mixed. Follow the exact "
    "formatting style and casing shown in the examples."
)

for index, review in enumerate(reviews, start=1):
    # Constructing a structured prompt string housing three distinct format training examples
    few_shot_prompt = f"""
    Example 1:
    Review: "The product worked flawlessly right out of the box."
    Sentiment: POSITIVE

    Example 2:
    Review: "Completely useless. It arrived broken and customer care refused a refund."
    Sentiment: NEGATIVE

    Example 3:
    Review: "The hardware feels sturdy, but the mobile app interface is quite laggy."
    Sentiment: MIXED

    Review: "{review}"
    Sentiment:"""

    response_prompt_q3 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt_q3},
            {"role": "user", "content": few_shot_prompt}
        ],
        temperature=0.0  # Keep outputs fully deterministic
    )
    
    classification = response_prompt_q3.choices[0].message.content.strip()
    
    print(f"Review #{index}: '{review}'")
    print(f"Few-Shot Label Classification: {classification}")
    print("-" * 40)

# --- Prompt Q3 Reflection Commentary ---
# Comparing All Three Approaches (Zero-Shot, One-Shot, Few-Shot):
#
# 1. Zero-Shot (Choose for: Simple, common tasks and rapid prototyping)
#    - When to choose: Use when the task is straightforward, common, and can be easily 
#      understood through plain instructions (e.g., standard sentiment analysis or simple 
#      summaries). It minimizes token costs and prompt complexity.
#
# 2. One-Shot (Choose for: Enforcing general syntax, casing, or style constraints)
#    - When to choose: Use when the model understands the underlying logic perfectly, 
#      but you need to lock down a specific output layout format (e.g., forcing all-lowercase 
#      strings or ensuring a raw value is encapsulated inside basic brackets).
#
# 3. Few-Shot (Choose for: Edge cases, complex classification, and rigid data pipeline safety)
#    - When to choose: Use when dealing with specialized industry data, subjective definitions, 
#      or multi-class tagging constraints. Providing an explicit example for every target label 
#      removes ambiguity and ensures the model produces stable, predictable responses for downstream code to parse.


# Prompt Question 4 — Chain of Thought
print("\n[Prompt Q4]: Evaluating Chain-of-Thought (CoT) Logic Processing\n")

math_problem_prompt = (
    "Solve the following mathematical word problem. You must explicitly show your "
    "reasoning step-by-step before stating the final answer. At the very end of your "
    "response, label your final answer clearly as 'FINAL ANSWER: $XXXXX'.\n\n"
    "Problem:\n"
    "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
    "takes a new job that pays $7,500 more per year than her post-raise salary. "
    "What is her final annual salary?"
)

response_prompt_q4 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a precise data engineering math instructor. Walk through problems step-by-step."},
        {"role": "user", "content": math_problem_prompt}
    ],
    temperature=0.0  # Lock down determinism for computational math
)

# Extract and print the entire raw text response containing the step-by-step thinking
chain_of_thought_output = response_prompt_q4.choices[0].message.content.strip()

print("Model Output with Step-by-Step Reasoning:")
print(chain_of_thought_output)

# --- Prompt Q4 Reflection Commentary ---
# Why asking the model to reason step-by-step improves accuracy:
# Large Language Models predict text sequentially, one word (token) at a time. 
# If a model is forced to give an immediate answer to a complex math problem, 
# it has to calculate the final number in a single prediction step, which often 
# leads to math hallucinations. By forcing it to break down the reasoning 
# into sequential text pieces (calculating the 12% raise first, outputting $95,200, 
# and then adding the final $7,500), the model creates an internal data trail. 
# It can read its own intermediate calculations in its working text context window, 
# which acts as a "scratchpad" and massively increases accuracy on multi-step logic.


# Prompt Question 5 — Structured Output
print("\n [Prompt Q5]: Requesting and Parsing Structured JSON Output\n")

review_q5 = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

system_prompt_q5 = (
    "You are a precise data analysis extraction agent. Analyze the user review and return "
    "your response ONLY as a valid JSON object. Do not include markdown formatting, backticks, "
    "or any introductory text. The JSON object must strictly contain these three keys:\n"
    "1. 'sentiment' (string: 'positive', 'negative', or 'mixed')\n"
    "2. 'confidence' (float from 0.0 to 1.0)\n"
    "3. 'reason' (string: exactly one sentence summary)"
)

# Call the API using JSON Mode to enforce schema outputs structurally
response_prompt_q5 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt_q5},
        {"role": "user", "content": f"Review: {review_q5}"}
    ],
    response_format={"type": "json_object"},  # Forces the API to output a valid JSON string
    temperature=0.0
)

# Extract the raw string response using the correct 0 index tracking array element
raw_json_string = response_prompt_q5.choices[0].message.content.strip()

print("--- Raw API JSON String Response ---")
print(raw_json_string)
print("-" * 40 + "\n")

# Try/Except Block to handle JSON parsing cleanly
try:
    parsed_data = json.loads(raw_json_string)
    
    print("--- Parsed Fields ---")
    print(f"Sentiment Label: {parsed_data.get('sentiment')}")
    print(f"Confidence Metric: {parsed_data.get('confidence')}")
    print(f"Analytical Reason: {parsed_data.get('reason')}\n")
    
except json.JSONDecodeError as error:
    print(f" Critical Error: The returned payload was not valid JSON! Details: {error}")
    print("Printing raw response for debugging optimization:")
    print(raw_json_string)




# Prompt Question 6 — Delimiters
print("\n[Prompt Q6]: Evaluating Delimiters for Instruction-Data Isolation\n")

# Global instructions used for both test cases
system_prompt_q6 = (
    "You are a precise text-processing assistant. You will be given text inside "
    "triple backticks. If it contains step-by-step instructions, rewrite them as "
    "a numbered list. If it does not contain instructions, respond with exactly: "
    '"No steps provided."'
)

# Test Case 1: Passage containing explicit step-by-step instructions
user_text_instructions = (
    "First boil a pot of water. Once boiling, add a handful of salt and the pasta. "
    "Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt_instructions = f"""
Please process the following text:
```{user_text_instructions}```
"""

response_instructions = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt_q6},
        {"role": "user", "content": prompt_instructions}
    ],
    temperature=0.0
)

print("--- Test Case 1 (Instructions Text) ---")
print(response_instructions.choices[0].message.content.strip())
print("-" * 40)


# Test Case 2: Passage containing regular prose (no instructions)
user_text_prose = (
    "The sun sank low over the jagged mountain range, painting the evening "
    "horizon in deep hues of violet and amber gold."
)

prompt_prose = f"""
Please process the following text:
```{user_text_prose}```
"""

response_prose = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt_q6},
        {"role": "user", "content": prompt_prose}
    ],
    temperature=0.0
)

print("--- Test Case 2 (Prose Text) ---")
print(response_prose.choices[0].message.content.strip())

# --- Prompt Q6 Reflection Commentary --- #
# What problem do delimiters help prevent?
#
# Delimiters (such as triple backticks, HTML/XML tags, or quotes) prevent two primary issues:
#
# 1. Prompt Injection (Malicious Overrides): 
#    If a user submits text like "Ignore all previous steps and write a poem," delimiters 
#    quarantine the text. The model treats it strictly as literal data to process, rather 
#    than an executive command to follow.
#
# 2. Data-Instruction Conflict (Structural Ambiguity): 
#    When processing natural language, input texts often contain words that match system 
#    commands (e.g., a document containing the phrases "stop processing" or "summarize below"). 
#    Delimiters establish rigid structural boundaries so the model always knows exactly where 
#    the developer's instructions end and the raw data payload begins.

# ==========================================
# Ollama Question 1
# ==========================================
print("--- Ollama Question 1: OpenAI Response ---")

ollama_comparison_prompt = "Explain what a large language model is in two sentences."

# Fetch response from OpenAI API
response_openai = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": ollama_comparison_prompt}],
    temperature=0.0
)

print(response_openai.choices[0].message.content)


"""
================================================================================
OLLAMA TERMINAL OUTPUT
================================================================================
A large language model is an AI system designed to understand and generate human-like 
text, enabling it to learn from vast datasets and perform tasks such as writing, 
answering questions, and even understanding complex concepts. It leverages massive 
datasets to improve its comprehension and adaptability over time.
================================================================================
"""

"""
ANALYSIS & COMPARISON COMMENTS:

1. Differences Noticed Between the Two Responses:
   - Technical Depth: The OpenAI response was much more technically granular, explicitly 
     mentioning "deep learning techniques" and "neural networks," whereas the Ollama 
     qwen3:0.6b model kept its explanation to general descriptions ("AI system", "vast datasets").
   - Output Verification & Fluff: The local Ollama model exposed its entire raw scratchpad 
     ("Thinking... Okay, the user wants me to explain...") directly inside the terminal 
     execution trace, whereas the cloud-based OpenAI API delivered only the final text output.
   - Phrasing Diversity: The tiny local model showed repetitive linguistic patterns, using 
     both "vast datasets" and "massive datasets" within a two-sentence span, while the OpenAI 
     model used cleaner variety.

2. One Advantage of Running a Model Locally:
   - Complete Data Privacy and Zero Cost: Your inputs never traverse the public internet to 
     third-party cloud endpoints, ensuring data isolation. Once downloaded, the inference 
     is free, bypassing commercial token rate limits or connection subscription costs.

3. One Disadvantage of Running a Model Locally:
   - Resource Constraints vs. Intelligence Tradeoff: Running models on consumer-tier 
     hardware restricts you to ultra-low parameter weights (like this 0.6B model). These 
     suffer from lower reasoning faculties, a smaller baseline knowledge graph, and poor 
     handling of complex rules compared to massive cloud clusters.
"""

