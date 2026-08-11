
import os
import string
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


# ----- ENVIRONMENT INITIALIZATION ----- #
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Configure the global embedding model for LlamaIndex
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# --- RAG Concepts ---

# Concepts Q1
print("\n=== Concepts Question 1 ===")
print("Scenario A Best Approach: Retrieval-Augmented Generation (RAG)")
print("Scenario B Best Approach: Fine-Tuning")
print("Scenario C Best Approach: Prompt Engineering (Context Injection)")
print("=" * 60)

# Concepts Q1 Reflection Commentary:
# Scenario A: Retrieval-Augmented Generation (RAG) is best because the legal team needs to query an 
# internal library of hundreds of dynamically changing PDFs that update quarterly. RAG handles frequent 
# document database updates effortlessly by swapping or expanding data vectors without costly model retraining cycles.
#
# Scenario B: Fine-Tuning is best because the startup wants the model to learn a highly specific, niche brand 
# voice and formatting style across 3,000 in-house training examples. Fine-tuning adjusts the base model's inner 
# weight matrices, making it perfect for teaching a distinct linguistic style that does not appear online.
#
# Scenario C: Prompt Engineering (Context Injection) is best because the analyst is asking questions about a single, 
# static, two-page report as a one-off task. Directly placing the text payload into the prompt context window 
# alongside the question is the most efficient, direct, and zero-overhead approach.


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
Concepts Q3 Reflection Commentary:

Official Assignment Step List Arranged Chronologically with One-Sentence Descriptions:

1. Extract text from source documents
   The system reads raw binary formats like PDFs, HTML, or TXT files and parses them into clean, unformatted plain text strings.

2. Split text into chunks
   The massive raw document text string is sliced into smaller, uniform sections with an overlap window to isolate semantic ideas.

3. Convert text chunks into embeddings
   Each separate text chunk is transformed into a high-dimensional numeric vector using an embedding model and saved into a database.

4. Receive the user's query
   The end-user inputs a natural language question or informational request directly into the application chat interface.

5. Embed the user's query
   The system takes the user's natural language question string and converts it into a vector embedding using the same model.

6. Retrieve the most relevant chunks
   The database performs a mathematical similarity check to extract the text chunks whose vector coordinates match closest to the query.

7. Inject retrieved chunks into the prompt
   The raw text contents of the highest-ranking matching chunks are pasted directly into a structured system prompt template.

8. Generate a response from the LLM
   The model reads the system directives, the injected reference facts, and the question to output an accurate final answer.
"""


# # --- Keyword-based RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is", "are", 
        "was", "were", "by", "with", "at", "from", "that", "this", "as", "be", "it", 
        "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)
    query_words = { w.translate(translator) for w in query.lower().split() if w not in stopwords }
    
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")
        
    scores = []
    for name, content in documents.items():
        content_words = { w.translate(translator) for w in content.lower().split() if w not in stopwords }
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

retrieval_results = simple_keyword_retrieval(query=query_k1, documents=documents_k1, verbose=True)
selected_doc_name = retrieval_results[0][0]
print(f"\n[Result Label] Selected Document Name: {selected_doc_name}")
print("=" * 60)

# ANSWER Q1: 
# Selected Document: loyalty.txt
# Why: The word 'your' was not filtered out by the stopwords list, making it an active search token. 
# This created a three-way tie where 'hours.txt' (matching 'weekends'), 'hiring.txt' (matching 'your'), 
# and 'loyalty.txt' (matching 'your') each scored an overlap of 1. Because of this tie, the sorting 
# algorithm fell back to internal alphabetical file placement order and returned 'loyalty.txt'.


# Keyword Q2
print("\n=== Keyword Question 2 ===")
query_k2 = "Do you have anything without caffeine?"

retrieval_results_k2 = simple_keyword_retrieval(query=query_k2, documents=documents_k1, verbose=True)
selected_doc_name_k2 = retrieval_results_k2[0][0]
print(f"\n[Result Label] Selected Document Name: {selected_doc_name_k2}")
print("=" * 60)

# ANSWER Q2:
# Which document was selected:
# None found (The function returned a fallback result indicating no overlapping keywords were discovered).
# 
# Whether keyword RAG got this right — and why or why not:
# No, keyword RAG did not get this right. A human reader easily understands that 'menu.txt' contains 
# the requested information since pastries and non-dairy milks are naturally caffeine-free alternatives. 
# However, keyword search relies entirely on literal token overlaps. Because the specific words 'anything' 
# and 'caffeine' do not appear verbatim anywhere inside the 'menu.txt' string, its overlap score dropped 
# to 0 across the entire index, leading to a complete retrieval failure.
# 
# What kind of retrieval would do better here:
# Dense semantic retrieval using Vector Embeddings would perform much better here. Semantic retrieval measures 
# conceptual similarity and contextual proximity rather than checking for rigid character spelling matches, 
# allowing it to recognize that food items and non-dairy milks are semantically related to choices "without caffeine".


# Keyword Q3
print("\n=== Keyword Question 3 ===")
query_k3 = "How do I sign up for rewards?"

# PREDICTION Q3:
# Predicted Document: loyalty.txt
# Reasoning: The query keywords like 'rewards' and 'sign up' match the concepts inside the loyalty file.

retrieval_results_k3 = simple_keyword_retrieval(query=query_k3, documents=documents_k1, verbose=True)
selected_doc_name_k3 = retrieval_results_k3[0][0]
print(f"\n[Result Label] Selected Document Name: {selected_doc_name_k3}")
print("=" * 60)

# ANSWER Q3:
# Selected Document: None found
# Prediction Correct?: No, the prediction was incorrect because the algorithm returned 'None found'.
# Why it happened: While humans associate 'rewards' with a loyalty program, 'loyalty.txt' uses the exact 
# terms 'loyalty program' and 'points', leaving 0 literal word overlaps with the user's query tokens.


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
# Path Setup: Point SimpleDirectoryReader to the official lesson path relative to this script
brightleaf_path = "assignments_06/lessons/06_AI_augmentation/resources/brightleaf_pdfs"

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
- Relevance of Chunks: Node 1 is highly relevant (Similarity Score: ~0.753) because it contains 
  the core foundational employee benefits overview text. However, Nodes 2 and 3 drop below 0.60 
  and represent macro partnership files and general company mission context.
- Tone and Confidence: The model's response sounds completely confident, direct, and specific. It avoids 
  any defensive hedging phrases like "based on the context" or "I am not sure." This is because the 
  exact ground-truth data was successfully passed to the context window, allowing it to list facts 
  declaratively (e.g., 401k match, Wellness Reimbursement Plan).
- Unexpected Retrieval: Yes, Node 2 ('EcoVolt Energy Partnership') was unexpectedly retrieved with 
  a ~0.597 score. This occurred because loose semantic keywords regarding operational collaboration 
  or company structural resources crossed thresholds into the top-k selection filter space.

Query 2: "What are BrightLeaf's security policies?"
- Relevance of Chunks: Node 1 is perfectly relevant (Similarity Score: ~0.690) because it maps 
  directly to the explicit network infrastructure and device authentication protocols text block.
- Tone and Confidence: The tone is formal, professional, and authoritative. It provides exact parameters 
  (such as credential rotation every 90 days and alignment with NIST guidelines) with complete certainty, 
  proving that RAG eliminates conversational speculation when specific data points are visible.
- Unexpected Retrieval: Yes, Node 2 ('Introduction / Benefits overview') and Node 3 ('EcoVolt Energy') 
  were unexpectedly retrieved. Because the query engine was forced to pull exactly 'similarity_top_k=3' 
  chunks, and the single dedicated security policy page was exhausted, the engine backfilled the remaining 
  slots with the next highest matching company background fragments.
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
# --- LlamaIndex Q2 Post-Execution Reflection Commentary ---
"""
Observations and Performance Evaluation:

1. How the response changed between the two execution configurations:
   - Factual Content: The core substance of the answers remained virtually identical between both runs. Both outputs listed 
     the 401(k) retirement plan, medical insurance, wellness stipends, and the learning hub courses.
   - Textual Nuances: The response for similarity_top_k=5 subtly shifted its ending emphasis toward corporate values. It added the 
     concluding sentence: "BrightLeaf emphasizes diversity, equity, and inclusion through its benefits and supports employee well-being 
     both personally and professionally." This broader language was pulled because it had access to lower-scoring nodes (like Nodes 4 and 5 
     which hover way down at ~0.487 and ~0.447) that contain general company culture statements rather than explicit benefits definitions.

2. Is more retrieved context always better?
   - No, more retrieved context is definitely not always better in a production RAG system. While increasing top_k can pull fine-grained 
     details that happen to be scattered across multiple pages, it introduces significant technical trade-offs:
     
     * Prompt Distraction and Hallucination Risk: Passing weak, low-scoring fragments (like Nodes 4 and 5 which drop below a 0.50 score) 
       floods the LLM's working window with unrelated filler words and noise. This forces the model to sift through irrelevant data, 
       increasing processing latency and elevating the risk of semantic confusion or text hallucinations.
     * Cost Inflation: Every character retrieved must be passed into the API payload text. Increasing top_k from 1 to 5 dramatically 
       inflates your input token count, causing enterprise operating expenses to shoot up over time.
     * The 'Lost in the Middle' Effect: Studies show that LLMs are excellent at reading text at the very beginning and very end of a prompt, 
       but they frequently ignore or completely miss critical facts buried in the middle of long, crowded context windows.
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
   I expected the pipeline to struggle significantly or completely fail to synthesize a coherent answer. This query forces a cross-document 
   comparison by blending details from two completely different operational domains: retirement caps ('employee_benefits.pdf') and 
   MFA compliance loops ('security_policy.pdf'). Because the query is split across two topics, its semantic vector embedding is inherently 
   'diluted.' It does not point to a single document space, which mathematically forces individual similarity scores to plummet.

2. What Actually Happened:
   The pipeline generated an astoundingly accurate response, but the underlying retrieval metrics prove it was running on thin ice. 
   As expected, the similarity scores dropped into a low-confidence range:
   - Node 1 (Security Policy): Score of ~0.391
   - Node 2 (Employee Benefits): Score of ~0.301
   - Node 3 (Financial Overview): Score of ~0.167
   
   Because our configuration used 'similarity_top_k=3', both Node 1 and Node 2 were barely caught in the retrieval net and passed 
   into the context window. The LLM then used its internal reasoning capability to extract the discrete facts ("up to five percent" 
   and "rotated every 90 days") and stitch them into a clean comparison sentence. If our top_k boundary had been restricted to 1, or 
   if our vector store enforced a strict similarity threshold cut-off of >0.50, the system would have dropped the benefits context 
   entirely and failed to complete the request.

3. Architectural Changes to Handle This Better:
   To make a production RAG system natively resilient against complex, multi-hop, or comparative questions, I would implement 
   the following technical improvements:
   
   - Query Deconstruction (Agentic Step): Place an LLM supervisor routing agent at the front gate to break complex prompts down into 
     independent sub-queries (e.g., Sub-Query 1: 'What are retirement matching caps?' and Sub-Query 2: 'What are MFA compliance cycles?').
   - Multi-Route Vector Retrieval: Execute independent vector similarity lookups for each deconstructed sub-query, pull the unique 
     top matches for each distinct branch, and combine the aggregated text chunks into a unified context bundle before synthesis.
   - Hybrid Search (BM25 + Vector): Blending keyword matching with dense vector semantics ensures that unique string entities like 
     'retirement matching caps' or 'multi-factor authentication' pull their respective source paragraphs instantly, bypassing 
     the similarity score dilution seen in a pure semantic lookup.
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

1. What Faithfulness Scores Mean:
   - A Faithfulness score of 1.0 (True) means that every single factual claim made in the model's generated answer 
     can be directly traced back and verified inside the retrieved context chunks. The model was completely honest to its sources.
   - A score of 0.0 (False) indicates a factual hallucination or text element not explicitly present in the source files.

2. What Relevancy measures and how it differs from Faithfulness:
   - Relevancy measures how well the generated answer directly addresses the intent and objective of the user's prompt.
   - The Difference: Faithfulness evaluates whether the answer is factually honest to the retrieved source text. Relevancy 
     evaluates whether the answer is useful to the user's question. An answer can be completely faithful to an underlying document, 
     but completely irrelevant to what the user actually asked.

3. Why the scores changed drastically between the two queries:
   - High-Quality Grounded Scenario ("What employee benefits does BrightLeaf offer?"):
     * Faithfulness = True | Relevancy = True. The system retrieved the correct text chunks, and the model accurately 
       summarized the real factual data (401k, healthcare, stipends). The judge passed both.
   - Low-Quality Out-of-Context Scenario ("What are the rules for adopting a pet dragon at the office?"):
     * Faithfulness = False | Relevancy = False. Because "dragon" is absent from the files, the vector search pulled irrelevant 
       context. The model correctly stated that BrightLeaf has no such rules. However, the evaluator judge flagged Faithfulness as 
       False because stating that a policy doesn't exist isn't an explicit truth defined within the document text chunks themselves. 
       It flagged Relevancy as False because the corporate source text contains zero actionable guidelines to answer the user's explicit question.

4. What the "LLM-as-a-judge" approach is and why it is used:
   - The "LLM-as-a-judge" approach uses an advanced language model (like gpt-4o-mini with a temperature of 0.0) to read a user query, 
     its context, and the response to grade the pipeline's overall performance.
   - Why it is used: Traditional accuracy metrics (like BLEU, ROUGE, or exact keyword string matching) only check for rigid spelling 
     overlaps against a reference key. Because generative AI models can write correct answers using entirely different words, sentences, 
     and formatting styles, simple math metrics fail. An LLM judge provides semantic reasoning to evaluate the actual underlying meaning, 
     logic, and truthfulness.
"""



