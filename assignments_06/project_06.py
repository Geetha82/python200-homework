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
docs_dir = Path("./groundwork_docs")

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

#Step 4 Reflection Comment Block:
# ------------------------------------------------------------------------------
# Reflection on Step 4 responses:

# 1. Did the assistant sound confident and accurate?Yes, the assistant sounded highly confident and completely accurate. 
# Everyanswer perfectly reflected the information inside the retrieved chunks
# (suchas extracting the exact 2018 founding year from our_story.txt and
# correctweekend hours from faq.txt) without adding any vague guesswork or defensive filler.

# 2. Did any of the answers surprise you?I was surprised by how accurately the semantic engine handled the dairy-free query.
# Even though the question asked broadly about "dairy-free milk options," the indexretrieved seasonal_specials.txt 
# with a high score of 0.7791, and the model perfectlypulled out 
# specific ingredients (oat, almond, and soy milk) and noted they are providedat no extra charge. 
# This showcases how vector indexing successfully matches conceptseven when text spans multiple contexts.


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

# Step 5 Evaluation & Diagnostics Comment Block:
# ------------------------------------------------------------------------------
# Explanation of Failure Mode:

# 1. What you asked and why you expected it to be hard:
# I asked if weekend breakfast customers earn extra loyalty points when bookinga catering wedding package. 
# This query is hard because it links keywords fromthree separate operational 
# aspects ("weekend breakfast", "loyalty points","catering wedding package") that do not intersect in any single document rule.

# 2. What went wrong — wrong retrieval, missing information, the model guessed anyway?This represents a 
# clear "Retrieval Gap" combined with an accidental correct guess by the LLM.The underlying documents contain 
# missing information—there is no text matching wedding points.Vector search retrieved 
# 'wholesale_catering.txt' (0.7827), 'faq.txt' (0.7597), and'seasonal_specials.txt' (0.7307) simply because those files 
# contain the individual fragments"catering," "weekend," and "loyalty." Because the text had no rule granting extra points,
# the model correctly deduced the answer was a negative, but it did so without actual proof.

# 3. Tone observation and uncertainty tracking when retrieval failed:
# The model's tone did not change at all. It remained fully confident, authoritative,and completely certain, 
# stating flatly: "Weekend breakfast customers do not earn extraloyalty points if they book a catering wedding package.
# " It did not mention that it lackeddirect supporting proof or that the context text was missing explicit wedding terms.

# 4. What this suggests about trusting AI-generated responses:
# This suggests that an AI response's professional, assertive tone is an 
# unreliable metric for truth.An engine can deliver a confident, logical answer while processing 
# irrelevant source nodesflawed by an active retrieval gap.5. What you would change about the system to
#  improve it:I would implement a programmatic similarity score guardrail or require an explicitsystem instruction 
# forcing the LLM to state "I cannot find information linking wedding packagesto loyalty structures" whenever 
# the retrieved fragments fail to address all compound clausesof a search parameter.

# ------------------------------------------------------------------------------


# --- Step 6 Reflection Commentary ---

# Observations and Performance Evaluation:

# 1. Code Density Comparison and the Value of a Framework:
#    - In a pure, manual, scratch-built RAG script, setting up a data pipeline—including file parsing, 
#      token splitting loops, structural window offsets, multi-threaded embedding API dispatches, 
#      and numpy matrix array dot-product computations—requires roughly 75 to 100 lines of complex math code.
#    - In contrast, the equivalent core LlamaIndex implementation in this project takes only 2 lines of code:
#      * Line 1: vector_index = VectorStoreIndex.from_documents(groundwork_docs)
#      * Line 2: query_engine = vector_index.as_query_engine(similarity_top_k=3)
#    - Value Proposition: This massive reduction proves that frameworks provide immense architectural value. They 
#      abstract away tedious, low-level technical infrastructure into modular, enterprise-ready components. This 
#      allows engineers to build, iterate, and deploy secure AI search systems at breakneck speed without wasting 
#      valuable time reinventing foundational algorithms.

# 2. Alternative High-Value Real-World Business Use Case:
#    - Use Case: An Internal Medical and Clinical Compliance Knowledge Base for a regional hospital network.
#    - Genuine Value: Instead of a coffee shop menu, the system ingest thousands of pages of frequently updated 
#      clinical treatment protocols, health insurance compliance mandates, standard pharmaceutical dosage charts, 
#      and state hospital safety regulations. 
#    - Operational Impact: On-duty nurses and emergency room doctors can instantly execute complex semantic queries 
#      to retrieve precise compliance guidelines or medication interaction boundaries during a crisis, completely 
#      bypassing the slow, stressful process of flipping through massive physical reference binders.

# 3. Failure Mode That RAG Cannot Fully Prevent:
#    - The "Lost in the Middle" and Synthesis Failure Mode. Even when retrieval works perfectly and the vector store 
#      successfully injects the exact ground-truth paragraph into the prompt context window, RAG cannot guarantee 
#      the LLM will analyze it correctly. 
#    - Vulnerabilities: Deep learning language models suffer from internal reasoning bugs, prompt alignment decay, 
#      and attention degradation when fed long contexts. If a vital piece of factual data is buried in the middle 
#      of a massive text chunk, the model may experience "prompt distraction." It might simply miss the target fact, 
#      misinterpret a subtle negating word (like "not" or "except"), or hallucinate an inaccurate synthesis anyway, 
#      despite having the complete, correct documentation sitting directly inside its working memory.


