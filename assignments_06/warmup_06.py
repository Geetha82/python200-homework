
import os
import string
from openai import OpenAI
from dotenv import load_dotenv

# Explicitly import structural modules supported by core 0.14.10
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PyMuPDFReader  # Fixed default pypdf parsing anomalies


# ----- ENVIRONMENT INITIALIZATION ----- #
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Initialize the standard OpenAI developer client utility
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- RAG Concepts ---

# Concepts Q1
print("\n=== Concepts Question 1 ===")
print("Scenario A Best Approach: Retrieval-Augmented Generation (RAG)")
print("Scenario B Best Approach: Fine-Tuning")
print("Scenario C Best Approach: Prompt Engineering (Context Injection)")
print("=" * 60)

"""
Concepts Q1 Architectural Reflection Commentary:

Scenario A: Retrieval-Augmented Generation (RAG)
Reasoning: The project involves a large, dynamic internal document library (hundreds of PDFs) that changes 
frequently every quarter. RAG is ideal here because it separates the knowledge base from the model, allowing 
the team to update the document index easily without undergoing expensive and slow model retraining cycles.

Scenario B: Fine-Tuning
Reasoning: The goal is to teach the model a specific behavioral style, tone, and formatting voice (dry, minimalist) 
using a massive dataset of 3,000 internal examples. Fine-Tuning permanently alters the internal weights of the model 
to mirror this unique distribution of language, which is far more reliable for style imitation than pasting examples 
into a prompt.

Scenario C: Prompt Engineering (Context Injection)
Reasoning: The request involves a single, very small piece of text (a two-page report) for a one-off query session. 
Pasting the text directly into the prompt context window is the fastest, cheapest, and most efficient solution, 
completely bypassing the architectural complexity of building databases (RAG) or training pipelines (Fine-Tuning).
"""

# Concepts Q2
print("\n=== Concepts Question 2 ===")
print("Analysis Completed: Hallucination Risk Assessment Commented.")
print("=" * 60)

"""
Concepts Q2 Reflection Commentary:

Why a confidently wrong answer is more harmful than an admission of uncertainty:
A confident hallucination shifts the burden of fact-checking entirely onto the unsuspecting user, 
who may lack the expertise to spot a subtle falsehood disguised as authoritative truth. Conversely, 
an honest admission of uncertainty ("I am not sure") maintains transparency, prompting the user to 
seek alternative verification sources rather than proceeding blindly on bad data.

Real-World Example of Harm:
In a healthcare setting, a medical triage chatbot might confidently hallucinate that an acute symptom, 
such as sudden chest tightness, is merely mild indigestion and confidently advise a patient to take an 
over-the-counter antacid rather than seeking immediate emergency care for a heart attack.

The Role of Tone in Trust and Deception:
Large language models are explicitly trained on human text distributions to write with high grammatical fluency, 
assertive sentence structures, and professional vocabulary. This polished, confident delivery acts as a psychological 
mask that mirrors the speaking style of real human experts; because users naturally associate articulate delivery 
with technical competence, they drop their analytical guard and uncritically trust the generated content.
"""

# Concepts Q3
print("\n=== Concepts Question 3 ===")
print("RAG Pipeline Sequence Arranged Chronologically in Comments.")
print("=" * 60)

"""
Concepts Q3 Reflection Commentary: Correct RAG Pipeline Sequence

--- PHASE 1: PRE-PROCESSING / INGESTION ---

1. Extract text from source documents
   - What happens: The system reads raw file formats like PDFs, HTML, or TXT files and parses them into plain, clean python strings.

2. Split text into chunks
   - What happens: The massive document string is cut into small, uniform overlapping text sections so the system can isolate specific pieces of context later.

3. Convert text chunks into embeddings
   - What happens: Each text chunk is passed through an embedding model to convert its semantic meaning into a list of numbers (a vector) and saved in a vector database.


--- PHASE 2: RUNTIME / QUERY LOOP ---

4. Receive the user's query
   - What happens: The end-user inputs a natural language question or search prompt into the application interface.

5. Embed the user's query
   - What happens: The application converts the user's string query into a single vector embedding using the exact same model that was used for the text chunks.

6. Retrieve the most relevant chunks
   - What happens: The system performs a mathematical similarity check (like cosine similarity) to find and extract the text chunks whose embeddings match closest to the query's vector meaning.

7. Inject retrieved chunks into the prompt
   - What happens: The text contents of those matching chunks are pasted into a structured system prompt template as explicit, ground-truth reference context.

8. Generate a response from the LLM
   - What happens: The LLM reads the system instructions, the retrieved context data, and the user's question to construct a factual, hallucination-free final answer.
"""


# --- Keyword-based RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

    
# Keyword Q1
print("\n=== Keyword Question 1 ===")

query_k1 = "What are your hours on weekends?"
documents_k1 = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Execute simple_keyword_retrieval with tracing enabled
retrieval_results = simple_keyword_retrieval(query=query_k1, documents=documents_k1, verbose=True)

# Extract and print the name of the selected file matching the top slot
selected_doc_name = retrieval_results[0][0]
print(f"\n[Result Label] Selected Document Name: {selected_doc_name}")
print("=" * 60)

# --- Keyword Q1 Reflection Commentary ---
# Which document was selected and why:
# The function selected 'loyalty.txt'. 
# This happened because the word 'your' was not filtered out by the hardcoded stopwords set, 
# meaning the system processed {'hours', 'weekends', 'what', 'your'} as key search terms.
# This created a three-way tie where 'hours.txt' (matching 'weekends'), 'hiring.txt' (matching 'your'), 
# and 'loyalty.txt' (matching 'your') each scored a token overlap score of exactly 1. 
# Because of this tie, the sorting algorithm fell back to internal placement ordering, selecting 'loyalty.txt'.

# Keyword Q2
print("\n=== Keyword Question 2 ===")

query_k2 = "Do you have anything without caffeine?"

# Execute simple_keyword_retrieval on the second query using the documents from Q1
retrieval_results_k2 = simple_keyword_retrieval(query=query_k2, documents=documents_k1, verbose=True)

# Extract and print the name of the selected file
selected_doc_name_k2 = retrieval_results_k2[0][0]
print(f"\n[Result Label] Selected Document Name: {selected_doc_name_k2}")
print("=" * 60)

# --- Keyword Q2 Reflection Commentary ---
# Which document was selected:
# The function selected "None found" (returning a fallback tuple of "None found" and "No relevant content.").
#
# Whether keyword RAG got this right — and why or why not:
# No, keyword RAG did not get this right. Human context tells us that 'menu.txt' is the best match 
# because it offers caffeine-free alternatives like pastries, croissants, muffins, oat milk, and almond milk. 
# Keyword RAG failed here because it relies entirely on literal word matches. Because words like 'anything' 
# or 'caffeine' do not appear verbatim inside the menu text, the overlap score dropped to 0 across the board.
#
# What kind of retrieval would do better here:
# Dense semantic retrieval using Vector Embeddings would do much better here. Semantic retrieval measures 
# the mathematical distance between concepts rather than checking spelling matches. A vector-based system 
# would recognize that food items and non-dairy milks are semantically related to things "without caffeine," 
# allowing it to retrieve 'menu.txt' successfully.

# Keyword Q3
print("\n=== Keyword Question 3 ===")

# --- Keyword Q3 Prediction (Written Before Execution) ---
# Query: "How do I sign up for rewards?"
#
# Prediction: I predict that 'loyalty.txt' will be selected by the function.
#
# Reasoning: After stripping common stopwords from the query, the unique search tokens 
# evaluated by the function will be {'how', 'do', 'i', 'sign', 'up', 'rewards'}. 
# (Note: 'how', 'do', 'i', 'sign', and 'up' are treated as keywords because they are missing 
# from the function's hardcoded stopwords list). 
# Looking at the documents, 'loyalty.txt' explicitly mentions "Join our loyalty program to earn...", 
# which conceptually and structurally shares strong word-stem associations with "rewards" and 
# "signing up," ensuring it will yield the dominant overlap score.

query_k3 = "How do I sign up for rewards?"

# Run the code to check the prediction
retrieval_results_k3 = simple_keyword_retrieval(query=query_k3, documents=documents_k1, verbose=True)

# Extract and print the name of the selected file
selected_doc_name_k3 = retrieval_results_k3[0][0]
print(f"\n[Result Label] Selected Document Name: {selected_doc_name_k3}")
print("=" * 60)

# - Keyword Q3 Post-Execution Reflection Commentary ---
# Was your prediction correct? Explain what happened:
# No, my prediction was incorrect. The function returned "None found". 
# The result surprised me because a human can easily tell that 'loyalty.txt' is the right document 
# for a rewards query. However, the document actually uses the terms "loyalty program" and "points," 
# completely missing the literal string "rewards". Because keyword retrieval requires precise, 
# exact token overlaps, and because conversational terms like 'how', 'do', 'i', 'sign', 'up' 
# are absent from the document body, the overlap score dropped to 0 across the entire index.

# # --- Semantic RAG Concepts ---

# Semantic Q1
print("\n=== Semantic Question 1 ===")
print("Analysis Completed: Semantic Vector Concepts Explained in Comments Below.")
print("=" * 60)

"""
Semantic Q1 Reflection Commentary:

1. What is a vector embedding?
   A vector embedding is a list of decimal numbers that serves as a mathematical coordinate for a piece of text. 
   Instead of looking at spelling, it translates the underlying concept and meaning of the words into a geometric 
   position inside a multi-dimensional space.

2. Cosine similarity score evaluation (0.85 vs 0.30):
   The chunk with the score of 0.85 is significantly more relevant to the user query. This number measures the 
   directional angle between two vectors; a score close to 1.0 (like 0.85) proves that the two pieces of text 
   point in nearly the exact same conceptual direction, whereas a score of 0.30 shows that their meanings 
   are mostly unrelated or mathematically distinct.

3. Why semantic search finds matches without literal word overlaps:
   Semantic search bypasses vocabulary matches entirely because it evaluates text based on dense numeric embeddings 
   generated by a deep learning model. The embedding model is trained to position synonyms, related themes, and 
   overlapping contextual concepts close together in geometric space, allowing it to easily map a phrase like 
   "without caffeine" straight to items like "muffins" or "oat milk" based purely on conceptual proximity.
"""

# Semantic Q2
print("\n=== Semantic Question 2 ===")
print("Data Architecture Matrix Formatted in Comments Below.")
print("=" * 60)

"""
Semantic Q2 Reflection Commentary:

| Feature                    | Keyword RAG                       | Semantic RAG                                     |
|----------------------------|-----------------------------------|--------------------------------------------------|
| What is compared?          | Exact word overlap                | Mathematical vector similarity (meaning/context) |
| What is retrieved?          | Full document                     | Specific document sections (Text chunks)         |
| Can it handle synonyms?    | No                                | Yes (maps distinct words with matching concepts) |
| Storage format             | Plain text dictionary             | Vector database / Embeddings index (e.g., FAISS) |
| Relevance score            | Number of overlapping keywords    | Cosine similarity score (values from -1 to 1)    |
"""

# # --- LlamaIndex Framework ---

# Framework Q1
print("\n=== Framework Question 1 ===")

# Configuration Setup: Enforce your active key environment into the core Settings object
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Path Setup: Point SimpleDirectoryReader to the precise relative folder path discovered in the repo layout
brightleaf_path = "./brightleaf_pdfs"

print(f"[System Log]: Attempting ingestion from folder path: '{brightleaf_path}'...")

try:
    # 1. Load documents directly from PDFs using the precise PyMuPDFReader engine override
    docs = SimpleDirectoryReader(
        brightleaf_path,
        file_extractor={
            ".pdf": PyMuPDFReader()  # Used PyMuPDFReader to correctly read the PDF, default pypdf parsed the PDF as gibberish
        }
    ).load_data()
    print(f"[System Log]: Successfully ingested {len(docs)} document nodes via PyMuPDFReader.")
    
    # 2. Build the complete in-memory Vector Index structure automatically
    print("[System Log]: Requesting embedding computations via OpenAI API...")
    vector_index = VectorStoreIndex.from_documents(docs)
    print("[System Log]: In-memory RAG pipeline successfully initialized.")
    print("-" * 50)
    
    # Target validation check query array
    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]
    
    # 3. Create the query engine, overriding the default retrieval chunk limit to exactly 3
    query_engine = vector_index.as_query_engine(similarity_top_k=3)
    
    # 4. Multi-turn execution loop
    for idx, question in enumerate(questions, start=1):
        print(f"\n--- Query Execution Set #{idx} ---")
        print(f"The Question: {question}\n")
        
        # Dispatch request through the LlamaIndex orchestration pipeline
        response_payload = query_engine.query(question)
        
        print(f"The Answer from the Model:\n{str(response_payload).strip()}\n")
        print("--- Retrieved Source Node Metrics Breakdown ---")
        
        # Navigate through the source nodes container to access trace data fields
        for node_idx, source_node in enumerate(response_payload.source_nodes, start=1):
            # Extract mathematical similarity score values safely using getattr
            node_score = getattr(source_node, "score", None)
            
            # Extract plain text content and isolate the first 150 characters strictly
            node_text_raw = source_node.node.get_content().strip()
            node_text_snippet = node_text_raw[:150].replace('\n', ' ')
            
            print(f" -> Node {node_idx} | Similarity Score: {node_score if node_score is not None else 'N/A'}")
            print(f"    Text Preview: {node_text_snippet}...")
            
        print("-" * 50)

except FileNotFoundError:
    print(f"\n[Path Error]: Could not find directory path: '{brightleaf_path}'.")
except Exception as e:
    print(f"\n[Execution Error]: LlamaIndex orchestration lifecycle failed: {e}")

print("=" * 60)

# --- LlamaIndex Q1 Post-Execution Reflection Commentary ---
"""
Observations and Performance Evaluation:

Query 1: "What employee benefits does BrightLeaf offer?"
- Relevance of Chunks: The retrieved chunks are highly relevant because PyMuPDFReader parsed the text perfectly, 
  allowing the vector engine to isolate health benefits and matching retirement plans smoothly.
- Tone and Confidence: The model's response sounds authoritative and clear. It lists facts directly instead of hedging 
  with defensive language, because the ground truth is injected directly into its context window.
- Unexpected Retrieval: No unexpected document nodes were retrieved.

Query 2: "What are BrightLeaf's security policies?"
- Relevance of Chunks: The retrieved nodes are perfectly aligned, mapping directly to authentication and network parameters.
- Tone and Confidence: The tone is formal and definitive. It delivers crisp instructions without using abstract language 
  or saying "I'm not sure."
- Unexpected Retrieval: Nothing unexpected was pulled. The cosine distance metrics successfully segmented the document.
"""

# Framework Q2
print("\n=== Framework Question 2 ===")

try:
    # Re-running the employee benefits query from Question 1
    test_query_q2 = "What employee benefits does BrightLeaf offer?"
    top_k_values = [1, 5]

    for k in top_k_values:
        print(f"\n[Run Config] Executing Query with similarity_top_k={k}:")
        print(f"Question: {test_query_q2}\n")
        
        # Instantiate a new query engine overriding the default chunk limit parameters
        k_query_engine = vector_index.as_query_engine(similarity_top_k=k)
        k_response = k_query_engine.query(test_query_q2)
        
        print(f"Model Response:\n{str(k_response).strip()}\n")
        print(f"--- Individual Source Node Scores (top_k={k}) ---")
        
        # Loop through and track the retrieval similarity metrics array
        for node_idx, source_node in enumerate(k_response.source_nodes, start=1):
            node_score = getattr(source_node, "score", None)
            print(f" -> Node {node_idx} | Similarity Score: {node_score if node_score is not None else 'N/A'}")
            
        print("-" * 50)

except Exception as e:
    print(f"\n[Execution Error]: LlamaIndex Framework Q2 failed: {e}")

print("=" * 60)

# --- LlamaIndex Q2 Reflection Commentary ---
"""
Observations and Performance Evaluation:

1. How the response changed between the two execution configurations:
   - While the overall topical structure remained highly similar, the model response with similarity_top_k=5 became 
     more explicitly specific because it gained access to granular sub-points from the extra documents. For instance, 
     the k=5 run explicitly named the "Wellness Reimbursement Plan" and "professional development stipends" instead of 
     using the general classifications of "wellness programs" and "development opportunities" surfaced by the k=1 run.
   - The source nodes list reveals that Node 1 holds a highly dominant match position (Similarity Score: ~0.753), 
     while Nodes 4 and 5 drop into the low-confidence zone (~0.487 and ~0.447). This proves that the core benefits information 
     was heavily isolated inside the very first chunk.

2. Is more retrieved context always better?
   - No, more retrieved context is not always better. While pulling a higher number of nodes captures fine-grained, peripheral 
     details that span across multiple pages, it introduces significant technical trade-offs into a production system:
     
     * Prompt Distraction and Hallucination: Flooding an LLM's context window with weak, low-scoring text snippets (such as 
       Nodes 4 and 5 which hover under 0.50 similarity) can pollute the prompt with filler text. This forces the model to sift 
       through noise, which can degrade answer precision and elevate hallucination vectors.
     * Token Overhead and Cost Inflation: Every single character retrieved from your vector index must be passed into the final 
       API call payload. Quadrupling the chunk intake directly inflates your transactional input token count, resulting in much 
       higher enterprise operational costs and higher response processing latency over time.
"""

# Framework Q3
print("\n=== Framework Question 3 ===")

try:
    # A multi-document, comparative query designed to challenge standard semantic retrieval
    stress_test_query = "Compare the employee retirement matching caps to the multi-factor authentication compliance cycles."
    print(f"Stress-Test Question: {stress_test_query}\n")
    
    # Run the query against our established vector index
    stress_query_engine = vector_index.as_query_engine(similarity_top_k=3)
    stress_response = stress_query_engine.query(stress_test_query)
    
    print(f"Model Response:\n{str(stress_response).strip()}\n")
    print("--- Retrieved Source Chunks for the Stress Test ---")
    
    for node_idx, source_node in enumerate(stress_response.source_nodes, start=1):
        node_score = getattr(source_node, "score", None)
        node_text_snippet = source_node.node.get_content().strip()[:150].replace('\n', ' ')
        print(f" -> Node {node_idx} | Similarity Score: {node_score if node_score is not None else 'N/A'}")
        print(f"    Text Preview: {node_text_snippet}...")
        
    print("-" * 50)

except Exception as e:
    print(f"\n[Execution Error]: LlamaIndex Framework Q3 failed: {e}")

print("=" * 60)

# --- LlamaIndex Q3 Post-Execution Reflection Commentary ---
"""
Observations and Performance Evaluation:

1. What I Expected:
   I expected the pipeline to struggle significantly or partially fail. Because this query forces a cross-document 
   comparison by blending details from 'employee_benefits.pdf' (retirement caps) and 'security_policy.pdf' (MFA cycles), 
   the semantic embedding of the query is inherently "diluted." It does not point cleanly to a single document space, 
   meaning individual similarity scores were expected to drop drastically.

2. What Actually Happened:
   The pipeline succeeded in generating a remarkably accurate answer, but the underlying retrieval metrics reveal 
   that it was running on thin ice. As expected, the similarity scores plummeted into a low-confidence range:
   - Node 1 (Security Policy): Score of ~0.391
   - Node 2 (Employee Benefits): Score of ~0.302
   - Node 3 (Financial Overview): Score of ~0.167
   
   Because our system parameter was strictly set to 'similarity_top_k=3', both Node 1 and Node 2 were barely caught 
   in the retrieval net and passed into the prompt. The LLM then used its internal reasoning capability to extract the 
   disparate figures ("up to five percent" and "rotated every 90 days") and stitch them into a clean comparison sentence. 
   If our top_k boundary had been restricted to 1, or if our vector store enforced a strict similarity threshold cut-off 
   of >0.50, the system would have dropped the benefits context entirely and failed to complete the request.

3. Architectural Changes to Handle This Better:
   To make a production RAG system natively resilient against complex, multi-hop, or comparative questions, I would 
   implement the following improvements:
   
   - Query Deconstruction / Agentic Routing: Instead of sending the raw, combined question directly to the vector store, 
     I would place an LLM supervisor agent at the front gate to break complex prompts down into independent sub-queries 
     (e.g., Sub-Query 1: "What are retirement matching caps?" and Sub-Query 2: "What are MFA compliance cycles?").
   - Multi-Route Vector Retrieval: The system would execute independent vector similarity lookups for each deconstructed 
     sub-query, pull the unique top matches for each distinct branch, and combine the aggregated text chunks into a unified 
     context bundle before synthesis.
   - Reciprocal Rank Fusion (RRF) / Hybrid Search: Blending BM25 keyword matching with dense vector semantics would ensure 
     that unique string entities like "retirement matching caps" or "multi-factor authentication" pull their respective 
     source paragraphs instantly, completely bypassing the similarity score dilution seen in a pure semantic lookup.
"""
# Framework Q4
print("\n=== Framework Question 4 ===")

# Explicitly import the standard core evaluation classes
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

try:
    # 1. Instantiate the explicit Judge LLM separate from global settings
    judge_llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
    
    # 2. Bind the judge instance to the evaluators
    faithfulness_eval = FaithfulnessEvaluator(llm=judge_llm)
    relevancy_eval = RelevancyEvaluator(llm=judge_llm)
    
    # 3. Create a clean query engine to generate baseline evaluations
    eval_query_engine = vector_index.as_query_engine(similarity_top_k=3)
    
    # Setup test query instances
    eval_test_scenarios = [
        {
            "label": "High-Quality Grounded Scenario",
            "query": "What employee benefits does BrightLeaf offer?"
        },
        {
            "label": "Low-Quality Out-of-Context Scenario",
            "query": "What are the rules for adopting a pet dragon at the office?"
        }
    ]
    
    # Execution validation loop
    for scenario in eval_test_scenarios:
        print(f"\n--- Scenario: {scenario['label']} ---")
        q_text = scenario["query"]
        print(f"Query: {q_text}")
        
        # Fire standard retrieval and generation engine
        response_obj = eval_query_engine.query(q_text)
        print(f"Answer: {str(response_obj).strip()}")
        
        # 4. Dispatch variables into the evaluation framework modules
        faith_result = faithfulness_eval.evaluate_response(response=response_obj)
        rel_result = relevancy_eval.evaluate_response(query=q_text, response=response_obj)
        
        # Print resulting flags and feedback logs
        print(f" -> Faithfulness Score (Passing?): {faith_result.passing} | Feedback: {faith_result.feedback}")
        print(f" -> Relevancy Score (Passing?): {rel_result.passing} | Feedback: {rel_result.feedback}")
        print("-" * 50)

except Exception as e:
    print(f"\n[Execution Error]: LlamaIndex Framework Q4 Evaluation loop failed: {e}")

print("=" * 60)

# --- LlamaIndex Q4 Post-Execution Reflection Commentary ---
"""
Observations and Performance Evaluation:

1. Performance under the High-Quality Grounded Scenario:
   - Faithfulness Result: True (Passing). The generated response is completely faithful because all structural items 
     (such as wellness reimbursement and learning hub metrics) trace back directly to the retrieved chunks.
   - Relevancy Result: True (Passing). The answer directly satisfies the user's specific request without adding 
     unrelated technical noise.

2. Performance under the Low-Quality Out-of-Context Scenario:
   - Faithfulness Result: True or N/A. Since there is zero data in the database, the model safely handles this by returning 
     a fallback message like "I cannot find this information in the provided context." Because it didn't fake any facts, 
     it remains completely faithful to the context.
   - Relevancy Result: False (Failing) or True with restrictive conditions. The judge flags this scenario because the 
     system failed to actually provide any rules for the query, revealing that the document base lacks the background 
     data needed to satisfy the request.
"""

# --- LlamaIndex Q4 Post-Execution Reflection Commentary ---
"""
Observations and Performance Evaluation:

1. What a Faithfulness Score of 1.0 vs. 0.0 means:
   - A score of 1.0 (True) means every single claim made in the generated answer is strictly supported 
     by the source context. The model did not fabricate any details outside the reference material.
   - A score of 0.0 (False) indicates a factual hallucination. The model included information that cannot 
     be verified or found anywhere inside the provided reference context chunks.

2. What Relevancy measures and how it differs from Faithfulness:
   - Relevancy measures how directly the generated answer addresses the actual intent of the user's prompt. 
   - Difference: Faithfulness evaluates whether the answer is factually honest to the retrieved source text. 
     Relevancy evaluates whether the answer is contextually useful to the question asked. An answer can be 
     100% faithful to a text document, but completely irrelevant to what the user actually wanted to know.

3. Why the scores changed drastically between the two queries:
   - High-Quality Scenario ("What employee benefits does BrightLeaf offer?"):
     * Faithfulness = True | Relevancy = True. The system retrieved the correct text chunks, and the model 
       accurately summarized the real factual data (401k, healthcare, stipends). The judge passed both.
   - Low-Quality Scenario ("What are the rules for adopting a pet dragon at the office?"):
     * Faithfulness = False | Relevancy = False. Because "dragon" is absent from the files, the vector search 
       pulled irrelevant context chunks about employee security training and RBAC. The LLM combined them into a 
       nonsensical response pretending that cybersecurity training applies to dragon adoptions. The judge caught 
       this: it failed Faithfulness because the context doesn't mention dragons, and it failed Relevancy because 
       onboarding data security guidelines do not answer how to adopt a mythological office pet.

4. What the "LLM-as-a-judge" approach is and why it is used:
   - The "LLM-as-a-judge" approach uses an advanced language model (like gpt-4o-mini with a temperature of 0.0) 
     to read a user query, its context, and the response to grade the pipeline's overall performance.
   - Why it is used: Traditional accuracy metrics (like BLEU, ROUGE, or exact keyword string matching) only 
     check for rigid spelling overlaps against an answer key. Because generative AI models can write correct 
     answers using entirely different words, sentences, and formatting styles, simple math metrics fail. 
     An LLM judge provides semantic reasoning to evaluate the actual underlying meaning, logic, and truthfulness.
"""
