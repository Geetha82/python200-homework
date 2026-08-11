import os
import string
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex


# STEP 1: INITIALIZATION & ENVIRONMENT SETUP
print(f"\nStep 1: Setup\n")

# Load environment configuration variables from the active environment file
load_dotenv()

# Verify and extract the active OpenAI API key from system variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("[Initialization Error]: OPENAI_API_KEY is missing from your .env configuration.")

print("[System Log]: OpenAI client successfully initialized with active API Key configuration.")

# Direct path definition to comply strictly with instruction parameters
docs_dir = Path("assignments_06/lessons/06_AI_augmentation/resources/groundwork_docs")

# Step 1 Mandate: Enforce path presence with a defensive assertion stop check
assert docs_dir.exists(), f"[Critical Error] Document directory not found: {docs_dir}"
print(f"[System Log]: Workspace path check passed. Groundwork documents directory found at: '{docs_dir}'")
print("=" * 70)


# STEP 2: LOAD THE DOCUMENTS VIA SIMPLEDIRECTORYREADER
print(f"\nStep 2: Load the Documents\n")

print("[System Log]: Initiating document ingestion loop from data directory...")

try:
    # 1. Instantiate the directory reader to parse text assets
    reader = SimpleDirectoryReader(input_dir=str(docs_dir))
    
    # 2. Extract files into raw data structures
    groundwork_docs = reader.load_data()
    
    # 3. Print the total count of documents loaded
    print(f"Total documents loaded: {len(groundwork_docs)}")
    print("-" * 70)
    
    # 4. Iterate over document nodes to isolate the metadata file name values
    for doc in groundwork_docs:
        # Extract the file_name key directly out of the metadata dictionary
        file_name = doc.metadata.get("file_name", "Unknown File")
        print(f"File Name: {file_name}")

except Exception as e:
    print(f"\n[Ingestion Error]: Failed to read or parse directory components: {e}")

print("=" * 70)


# STEP 3: BUILD THE INDEX AND QUERY ENGINE
print(f"\nStep 3: Build the Index and Query Engine\n")

print("[System Log]: Initiating Vector Space embedding conversion...")

try:
    # 1. Build the index from the loaded document structures
    # This step executes network requests to calculate your text embeddings
    vector_index = VectorStoreIndex.from_documents(groundwork_docs)
    
    # 2. Create the query engine, overriding the top_k parameter to exactly 3
    query_engine = vector_index.as_query_engine(similarity_top_k=3)
    
    # 3. Print the exact requested assignment confirmation message
    print("Index built successfully. Ready to answer questions.")

except Exception as e:
    print(f"\n[Index Error]: Failed to construct vector index: {e}")

print("=" * 70)

# STEP 4: QUERY THE ASSISTANT via AGGREGATION LOOP
print(f"\nStep 4: Query the Assistant\n")

print("[System Log]: Dispatching target query execution verification array...")

# The explicit question tracking list required by your project criteria
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

try:
    # Multi-turn automated query inspection loop
    for idx, q_text in enumerate(questions, start=1):
        print(f"\n--- Query Execution Set #{idx} ---")
        print(f"Question: {q_text}\n")
        
        # Dispatch query to the active LlamaIndex RAG wrapper engine
        response_payload = query_engine.query(q_text)
        
        print(f"Answer from the model:\n{str(response_payload).strip()}\n")
        print("--- Top Retrieved Source Node ---")
        
        # Verify if any text context nodes were captured in the retrieval net
        if response_payload.source_nodes:
            # Isolate the top node (index 0) from the similarity list
            top_node = response_payload.source_nodes[0]
            
            # Extract document filename cleanly out of the metadata dictionary container
            doc_name = top_node.node.metadata.get("file_name", "Unknown File")
            
            # Extract the raw mathematical directional proximity score value safely
            similarity_score = getattr(top_node, "score", None)
            
            # Extract plain text contents, clean formatting breaks, and slice down to exactly 200 characters
            raw_text = top_node.node.get_content().strip()
            text_snippet = raw_text[:200].replace('\n', ' ')
            
            print(f"Document Name   : {doc_name}")
            print(f"Similarity Score: {similarity_score if similarity_score is not None else 'N/A'}")
            print(f"Chunk Text Clip : {text_snippet}...")
        else:
            print("[Warning]: No context source fragments were retrieved for this query string.")
            
        print("-" * 70)

except Exception as e:
    print(f"\n[Execution Error]: Evaluation loop execution failed: {e}")

print("=" * 70)

# --- Step 4 Post-Execution Reflection Commentary ---
"""
Observations and Performance Evaluation:
1. Did the assistant sound confident and accurate?
   Yes, the assistant sounded exceptionally confident, accurate, and professional throughout all five runs. 
   Because its prompt context was heavily grounded with high-confidence semantic chunks from the local files, 
   it stated specific facts directly (such as weekend hours being '8:00 AM to 5:00 PM' from faq.txt or points expiring at 
   '100 points' for a free drink) without using vague hedging phrases like "I think" or "based on the text."

2. Did any of the answers surprise you?
   The behavior of Query #2 ('Do you offer any dairy-free milk options?') provided an insightful 
   revelation. I expected the query engine to primarily pull 'menu.txt' to answer an ingredient question. Instead, 
   the vector model isolated 'seasonal_specials.txt' as the top retrieved source node with a high similarity score 
   of ~0.780. Even though the top chunk's text preview clipped right at the word 'Dairy-fre...', the engine utilized the 
   'similarity_top_k=3' context buffer to pass enough background details to the LLM, resulting in an accurate and 
   helpful answer explicitly enumerating oat milk, almond milk, and soy milk. 
   Additionally, Query #3 matched 'faq.txt' with a score of ~0.770 because 'faq.txt' contains consolidated business answers 
   that map strongly onto the user's vocabulary structure. Meanwhile, Query #4 accurately targeted 'our_story.txt' with 
   a high similarity score of ~0.904, enabling the LLM to extract the precise foundation background of Maya Torres and Sam Okafor.
"""

# STEP 5: FIND A FAILURE (STRESS-TEST INQUIRY)
print(f"\nStep 5: Find a Failure\n")

# A combined, speculative query designed to stretch and break semantic routing boundaries
failure_test_query = "Do our weekend breakfast customers earn extra loyalty points if they book a catering wedding package?"
print(f"Stress-Test Question: {failure_test_query}\n")

try:
    # Run the query using the established query engine configuration
    failure_response = query_engine.query(failure_test_query)
    
    print(f"Model Response:\n{str(failure_response).strip()}\n")
    print("--- All Three Retrieved Source Nodes for the Stress Test ---")
    
    # Iterate and print metrics for all 3 retrieved chunks to inspect the engine's performance
    for idx, source_node in enumerate(failure_response.source_nodes, start=1):
        node_score = getattr(source_node, "score", None)
        doc_name = source_node.node.metadata.get("file_name", "Unknown File")
        
        # Isolate text content and truncate smoothly down to exactly 200 characters
        raw_content = source_node.node.get_content().strip()
        text_snippet = raw_content[:200].replace('\n', ' ')
        
        print(f" -> Node {idx} | Document: '{doc_name}' | Similarity Score: {node_score if node_score is not None else 'N/A'}")
        print(f"    Text Preview: {text_snippet}...")
        print("-" * 50)

except Exception as e:
    print(f"\n[Execution Error]: LlamaIndex Framework Step 5 failed: {e}")

print("=" * 70)

# --- Step 5 Post-Execution Reflection Commentary ---
"""
Observations and Performance Evaluation:
1. What I Asked and Why I Expected it to be Hard:
   The query targets nonexistent policy intersections. It requires scanning multiple completely distinct 
   functional domains—wedding packages, weekend hours, and loyalty structures—and forces the engine to verify 
   if an explicit point multiplier exists for a specific customer sub-type.

2. What Went Wrong (Retrieval vs. Generation Analysis):
   - Retrieval Failure: The semantic embedding search suffered from severe keyword dilution. Because the question contained 
     words like "catering," "wedding," and "weekend," the vector database fetched 'wholesale_catering.txt' as Node 1 (Score: ~0.785), 
     'faq.txt' as Node 2 (Score: ~0.762), and 'seasonal_specials.txt' as Node 3 (Score: ~0.733). It failed to retrieve any 
     dedicated loyalty program reference documents altogether.
   - Generation Failure (The Model Guessed): Even though the actual information was completely missing from the retrieved context, 
     the model did not flag the data absence. Instead, it hallucinated a definitive negative assertion: "Weekend breakfast 
     customers do not earn extra loyalty points if they book a catering wedding package." While this claim happens to be 
     true for the business, the LLM fabricated this fact on the spot because nothing in the active context fragments 
     supported or denied it.

3. Analysis of Tone and Certainty:
   The model's tone did not change whatsoever; it remained completely direct, confident, and declarative despite being wrong. 
   It did not use defensive phrases like "based on the text, I am unsure" or hedge its stance. It presented a complete guess 
   with the exact same authoritative tone used for fully grounded factual responses.

4. What this Suggests About Trusting AI-Generated Responses:
   This proves that a model's confidence is never a reliable indicator of factual truth. Because LLMs are trained to generate 
   smooth, plausible-sounding linguistic patterns, they will state completely fabricated hallucinations or blind guesses 
   with absolute authority. Without explicit constraints or validation layers, a system can seamlessly mislead users while 
   sounding entirely certain of its claims.

5. Architectural Changes to Improve the System:
   - Strict Context Boundary Guarding: Modify the base system prompt instructions to state: "If the provided context fragments 
     do not explicitly state or confirm a policy, you must respond with: 'I am sorry, but that information is not available.'"
   - Similarity Score Thresholding: If a query does not produce context nodes above a certain similarity threshold (e.g., >0.82), 
     the system should reject the generation cycle outright.
"""

# STEP 6: FINAL ARCHITECTURAL REFLECTION

# --- Step 6 Reflection Commentary ---
"""
Observations and Performance Evaluation:

1. Code Density Comparison and the Value of a Framework:
   - In a pure, manual, scratch-built RAG script, setting up a data pipeline—including file parsing, 
     token splitting loops, structural window offsets, multi-threaded embedding API dispatches, 
     and numpy matrix array dot-product computations—requires roughly 75 to 100 lines of complex math code.
   - In contrast, the equivalent core LlamaIndex implementation in this project takes only 2 lines of code:
     * Line 1: vector_index = VectorStoreIndex.from_documents(groundwork_docs)
     * Line 2: query_engine = vector_index.as_query_engine(similarity_top_k=3)
   - Value Proposition: This massive reduction proves that frameworks provide immense architectural value. They 
     abstract away tedious, low-level technical infrastructure into modular, enterprise-ready components. This 
     allows engineers to build, iterate, and deploy secure AI search systems at breakneck speed without wasting 
     valuable time reinventing foundational algorithms.

2. Alternative High-Value Real-World Business Use Case:
   - Use Case: An Internal Medical and Clinical Compliance Knowledge Base for a regional hospital network.
   - Genuine Value: Instead of a coffee shop menu, the system ingest thousands of pages of frequently updated 
     clinical treatment protocols, health insurance compliance mandates, standard pharmaceutical dosage charts, 
     and state hospital safety regulations. 
   - Operational Impact: On-duty nurses and emergency room doctors can instantly execute complex semantic queries 
     to retrieve precise compliance guidelines or medication interaction boundaries during a crisis, completely 
     bypassing the slow, stressful process of flipping through massive physical reference binders.

3. Failure Mode That RAG Cannot Fully Prevent:
   - The "Lost in the Middle" and Synthesis Failure Mode. Even when retrieval works perfectly and the vector store 
     successfully injects the exact ground-truth paragraph into the prompt context window, RAG cannot guarantee 
     the LLM will analyze it correctly. 
   - Vulnerabilities: Deep learning language models suffer from internal reasoning bugs, prompt alignment decay, 
     and attention degradation when fed long contexts. If a vital piece of factual data is buried in the middle 
     of a massive text chunk, the model may experience "prompt distraction." It might simply miss the target fact, 
     misinterpret a subtle negating word (like "not" or "except"), or hallucinate an inaccurate synthesis anyway, 
     despite having the complete, correct documentation sitting directly inside its working memory.
"""

