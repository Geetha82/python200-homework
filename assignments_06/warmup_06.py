from dotenv import load_dotenv
import os
import math
import re
import string
from typing import List, Dict, Any, Tuple

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Initialize environment validation block
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# ==============================================================================
# --- RAG Concepts ---
# ==============================================================================

# Concepts Q1: Scenario Analysis for LLM Augmentation Strategies
# ------------------------------------------------------------------------------
# Scenario A: Legal team internal policy library (hundreds of PDFs updated quarterly)
# Best Approach: Retrieval-Augmented Generation (RAG)
# Reasoning: The library consists of hundreds of frequently updated documents, which is too vast for simple context injection and too dynamic for static fine-tuning. RAG connects a dynamic data store directly to a frozen LLM, providing scalable updates and crucial audit trails.
#
# Scenario B: Startup product copy in a specific brand voice (3,000 examples)
# Best Approach: Fine-Tuning
# Reasoning: The objective here is to train the model on nuanced behavior, style, and tone rather than factual information retrieval. Given a large, consistent dataset of 3,000 examples, fine-tuning modifies the model's internal weights to consistently match this dry, minimalist brand identity.
#
# Scenario C: Data analyst asking questions about a single two-page report
# Best Approach: Prompt Engineering (Context Injection)
# Reasoning: A single two-page document has a minimal token count that easily fits directly into the native context window of modern LLMs. Building an external vector database pipeline or retraining weights is unnecessary overhead for a quick, one-off analytical task.
# ------------------------------------------------------------------------------

print("\n# Concepts Q1: Evaluation of Augmentation Architectures")
strategies_summary = """
1. Context Injection:
   - Pros: Zero training cost, highly accurate for provided text, instant updates.
   - Cons: Bound by context window limits; linear cost growth with input token volume.
2. Fine-Tuning:
   - Pros: Optimizes formatting, style, tone, and deep domain-specific vocabulary rules.
   - Cons: Computationally expensive; slow iteration cycle; prone to factual hallucinations.
3. Retrieval-Augmented Generation (RAG):
   - Pros: Scalable to massive external enterprise corpuses; verifiable with audit trails/citations.
   - Cons: Relies heavily on retrieval accuracy; higher orchestration complexity.
"""
print(strategies_summary.strip())

print("\n# Scenario Recommendations Log:")
print("  Scenario A -> Best Approach: RAG (Handles hundreds of dynamically changing quarterly PDFs with strict traceability).")
print("  Scenario B -> Best Approach: Fine-Tuning (Alters behavioral stylistic weight parameters using 3,000 target examples).")
print("  Scenario C -> Best Approach: Prompt Engineering (Direct context injection for a tiny two-page static report).")

# Concepts Q2: The Harm of Confident Hallucinations and the Role of Tone
# ------------------------------------------------------------------------------
# 1. Why a confidently wrong answer is more harmful than "I am not sure":
#    An explicit admission of uncertainty ("I am not sure") serves as a transparent guardrail, prompting the user to perform independent verification. A confidently wrong answer, however, disarms human skepticism, inducing users to bypass fact-checking and execute decisions based on fundamentally flawed premises.
#
# 2. Real-world example of harm:
#    In a clinical setting, an AI diagnostic assistant might confidently hallucinate that a highly malignant skin lesion is completely benign and recommend simple topical ointment. This false confidence could cause a medical professional or patient to delay critical biopsy and oncological treatment, resulting in severe physical harm or accelerated disease progression.
#
# 3. Impact of response tone on trust:
#    The linguistic structure and tone of an output dictate human trust much more than the actual underlying data validity. When an LLM uses authoritative phrases, structured assertions, and definitive syntax, it maps directly onto human cognitive biases that equate professional fluency with expertise. This mismatch makes a hallucinated answer uniquely dangerous, as it perfectly mimics the expression of absolute truth.
# ------------------------------------------------------------------------------

print("\n# Concepts Q2: Analysis of Confident Hallucinations")
hallucination_analysis = """
- Harm Threshold: "I am not sure" preserves human vigilance; confident errors systematically bypass human skepticism.
- Clinical/Real-World Example: A medical diagnostic bot misinterpreting a critical symptom as benign, delaying life-saving medical care.
- Tone vs. Trust: Fluent, highly authoritative phrasing mirrors genuine expertise, exploiting linguistic biases to mask baseline statistical uncertainty.
"""
print(hallucination_analysis.strip())


# Concepts Q3: RAG Pipeline Step Ordering
# ------------------------------------------------------------------------------
# Original out-of-order list:
# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query"
# ]
#
# Correct Ordered Sequence:
# 1. Extract text from source documents
#    - Raw file formats like PDFs or HTML are parsed to extract clean raw text data.
# 2. Split text into chunks
#    - Large documents are broken down into smaller, bounded text passages to respect LLM context windows and isolate specific concepts.
# 3. Convert text chunks into embeddings
#    - Text segments are converted into multi-dimensional dense numerical vectors using an embedding model to capture semantic meaning.
# 4. Receive the user's query
#    - The pipeline captures the incoming natural language question or prompt submitted by the user.
# 5. Embed the user's query
#    - The user's query is converted into a vector embedding using the identical embedding model to ensure geometric alignment.
# 6. Retrieve the most relevant chunks
#    - The system calculates mathematical similarity (e.g., cosine similarity) between the query vector and chunk vectors to fetch top-k matches.
# 7. Inject retrieved chunks into the prompt
#    - The retrieved high-scoring text passages are inserted into a prompt template alongside the user's question as background context.
# 8. Generate a response from the LLM
#    - The augmented context prompt is sent to the LLM to synthesize a grounded, accurate answer without relying on stale internal data.
# ------------------------------------------------------------------------------

print("\n# Concepts Q3: Ordered RAG Pipeline Lifecycle")
rag_steps_ordered = """
1. Extract text from source documents
2. Split text into chunks
3. Convert text chunks into embeddings
4. Receive the user's query
5. Embed the user's query
6. Retrieve the most relevant chunks
7. Inject retrieved chunks into the prompt
8. Generate a response from the LLM
"""
print(rag_steps_ordered.strip())


# --- Keyword-based RAG ---
def simple_keyword_retrieval(query: str, documents: Dict[str, str], verbose: bool = True) -> List[Tuple[str, str]]:
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is", "are", 
        "was", "were", "by", "with", "at", "from", "that", "this", "as", "be", 
        "it", "its", "their", "they", "we", "you", "our", "your"
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
print("\n# Keyword Q1: Document Selection Evaluation")

q1_query = "What are your hours on weekends?"
q1_documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Execute keyword search with verbose trace diagnostics activated
retrieval_results = simple_keyword_retrieval(q1_query, q1_documents, verbose=True)

# Isolate and print only the designated file key name
selected_doc_name = retrieval_results[0][0]
print(f"\nSelected Document Name: {selected_doc_name}")


# Keyword Q1 Auxiliary Utilities: Character-Based Sliding Window Chunking Implementation
def chunk_text_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits an input text document into manageable slices using a character-based window.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        
    chunks = []
    start = 0
    effective_step = chunk_size - chunk_overlap
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += effective_step
        
        # Break constraint guarding against infinite termination issues
        if start >= len(text) or effective_step <= 0:
            break
            
    return chunks

print("\n# Keyword Q1 Template Check: Testing Text Chunking Utility")
sample_doc = "Python LLM workflows are powerful. Retrieval-Augmented Generation adds private context windows."
generated_chunks = chunk_text_by_chars(sample_doc, chunk_size=35, chunk_overlap=10)
for idx, text_slice in enumerate(generated_chunks):
    print(f"  Chunk {idx + 1}: '{text_slice}'")

# Keyword Q1 Explanation Comment Block:
# ------------------------------------------------------------------------------
# When matched against the dataset corpus, "hours.txt" is selected because it is 
# the only document with a positive intersection score (overlap=1), catching the 
# token 'weekends'. All other text documents ("menu.txt", "hiring.txt", 
# "loyalty.txt") yield an intersection score of exactly 0 since they share 
# no non-stopword tokens with the filtered query array.
# ------------------------------------------------------------------------------

# Keyword Question 2
print("\n# Keyword Question 2: Running a No-Overlap Search Vector Query")

q2_query = "Do you have anything without caffeine?"
# Uses the exact same document dictionary corpus from Keyword Q1

# Execute keyword search using our shared documents corpus dictionary
q2_results = simple_keyword_retrieval(q2_query, q1_documents, verbose=True)

selected_q2_doc = q2_results[0][0]
print(f"\nSelected Document Name: {selected_q2_doc}")

# Keyword Question 2 Explanation Comment Block:
# ------------------------------------------------------------------------------
# Explanation:
# 1. Which document was selected:
#    No actual document was selected. The system returned a fallback sentinel value 
#    of "None found" associated with a placeholder explanation string.
#
# 2. Whether keyword RAG got this right — and why or why not:
#    Technically, the keyword parser followed its deterministic token rules correctly, 
#    but from a functional user-experience perspective, it completely failed. The 
#    filtered query tokens were ['anything', 'caffeine', 'without'] (or including 
#    'do'/'have' if they are left unmapped by a strict stopword check). Because 
#    none of these exact word strings literally appear inside "menu.txt", the mathematical 
#    intersection score for every file was exactly 0. The system failed because it has 
#    no inherent understanding that items listed in "menu.txt"—such as "espresso", 
#    "lattes", and "cold brew"—are conceptually and chemically laden with caffeine.
#
# 3. What kind of retrieval would do better here:
#    A semantic retrieval system (such as dense vector embeddings processed via 
#    cosine similarity) would perform vastly better. Semantic frameworks convert text 
#    strings into geometric vectors inside a shared concept space. In that multi-dimensional 
#    space, words like "caffeine" align closely with "espresso", "latte", and "coffee", 
#    allowing a retriever to fetch "menu.txt" based on conceptual context, even if the 
#    exact word "caffeine" is entirely missing from the document's body text.
# ------------------------------------------------------------------------------

# Keyword Question 3: Prediction and Pre-Execution Analysis
# ------------------------------------------------------------------------------
# Prediction: 
# "loyalty.txt" will be selected.
#
# Reasoning:
# When the query "How do I sign up for rewards?" is evaluated, common structural 
# stopwords ("how", "do", "i", "for") are eliminated. This leaves behind two 
# clean search tokens: ['rewards', 'sign'].
#
# Let's inspect our documents for these filtered tokens:
# - menu.txt: Contains no references to signing up or rewards programs. (Overlap = 0)
# - hours.txt: Contains store operational details. (Overlap = 0)
# - hiring.txt: Contains job opening keywords ("hiring", "baristas", "resume"). (Overlap = 0)
# - loyalty.txt: Contains text outlining the reward program mechanics ("loyalty program", 
#   "earn one point per dollar spent", "Redeem 100 points"). 
#
# While "loyalty.txt" describes a customer incentives framework, notice that the 
# exact string token "rewards" does not literally appear in its text, nor does 
# the phrase "sign up". Therefore, despite being the most conceptually relevant 
# file, the keyword algorithm will score an absolute intersection value of 0 across 
# ALL documents. As a result, the function will trigger its fallback condition 
# and return "None found".
# ------------------------------------------------------------------------------

print("\n# Keyword Question 3: Document Match Prediction Validation Loop")

q3_query = "How do I sign up for rewards?"
q3_results = simple_keyword_retrieval(q3_query, q1_documents, verbose=True)
print(f"\nSelected Document Name: {q3_results[0][0]}")

# Keyword Question 3 Post-Execution Post-Mortem Comment Block:
# ------------------------------------------------------------------------------
# Post-Execution Analysis:
# The prediction that no correct document would be found was entirely correct! 
# The function executed and returned 'None found' as expected.
#
# Observation on Filtered Tokens:
# In the provided terminal execution logs, the filtered query output printed 
# ['do', 'how', 'i', 'rewards', 'sign', 'up']. This reveals that the local copy 
# of the 'simple_keyword_retrieval' function used a specific stopwords dictionary 
# that did not include operational or question tokens like "do", "how", "i", or "up".
#
# What happened:
# Even though the system retained these structural query words, it still produced 
# an absolute overlap score of 0 across all text entries. This occurred because 
# "loyalty.txt" describes the mechanics using alternative expressions—specifically 
# using phrases like "Join our loyalty program to earn one point..."—without ever 
# containing the exact string words "rewards" or "sign up". This test clearly 
# demonstrates that keyword matching is highly sensitive to token choices; if a 
# user describes a concept using synonyms or slightly different grammar, the 
# system fails entirely, further proving why semantic embeddings are vital.
# ------------------------------------------------------------------------------

# ==============================================================================
# --- Semantic RAG Concepts ---
# ==============================================================================

# Semantic Question 1: Conceptual Understanding of Vector Spaces (Beginner Level)
# ------------------------------------------------------------------------------
# 1. What is a vector embedding?
#    Think of a vector embedding as giving a piece of text its own unique GPS coordinates 
#    on a massive, multi-dimensional "map of meanings." Instead of using latitude and 
#    longitude, a computer uses a long list of numbers to plot the text so that ideas 
#    with similar meanings automatically end up sitting right next to each other on the map.
#
# 2. Relative Relevance & Cosine Similarities Interpretation:
#    The chunk with a score of 0.85 is much more relevant than the one with 0.30. 
#    In this system, a score closer to 1.0 means the text vectors are pointing in almost 
#    the exact same direction (like two cars driving down the same highway), meaning they 
#    are closely related. A low score like 0.30 means the texts are pointing in completely 
#    different directions and share very little in common.
#
# 3. Why Semantic Search transcends Literal Word Intersections:
#    Semantic search works because it looks at the *ideas* behind the words rather than 
#    matching letters character-by-character. Because the AI model was trained on millions 
#    of sentences, it already knows that a query asking about "rewards" is looking for the 
#    same concept as a document talking about a "loyalty program"—so it plots them close 
#    together on our meaning map, letting the system connect them even if they don't share 
#    a single exact word.
# ------------------------------------------------------------------------------

# Semantic Question 2: Architectural Framework Differences Summary
# ------------------------------------------------------------------------------
# | Feature                  | Keyword RAG                     | Semantic RAG                                  |
# |--------------------------|---------------------------------|-----------------------------------------------|
# | What is compared?        | Exact word overlap              | Conceptual meaning and contextual ideas       |
# | What is retrieved?       | Full document                   | Small, specific bounded text passages (chunks)|
# | Can it handle synonyms?  | No                              | Yes                                           |
# | Storage format           | Plain text dictionary           | Multi-dimensional numerical vector store      |
# | Relevance score          | Number of overlapping keywords  | Geometric similarity score (e.g., Cosine Sim) |
# ------------------------------------------------------------------------------

# ==============================================================================
# --- LlamaIndex ---
# ==============================================================================

# LlamaIndex Global Architecture Configuration
print("\n# LlamaIndex: Initializing Global Project Configurations")

# 1. Configure the global embedding model layer to text-embedding-3-small
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 2. Set chunk parameters to split document pages predictably
Settings.chunk_size = 512
Settings.chunk_overlap = 32
print("  LlamaIndex settings bound to active application memory contexts.")


# LlamaIndex Dataset Parsing and Vector Construction
print("\n# LlamaIndex: Parsing Local PDF Data Stores into Vectors")

pdf_directory_path = "./brightleaf_pdfs"

# Early Guard Clause: Eradicate nested blocks and satisfy IDE static code checking
if not os.path.exists(pdf_directory_path) or not os.path.isdir(pdf_directory_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: Target directory path '{pdf_directory_path}' was not found.\n"
        f"Please verify your folder mapping configuration settings layout."
    )

print(f"  Target path location verified: '{pdf_directory_path}' found.")

# Ingest text strings straight from the local folder
reader = SimpleDirectoryReader(input_dir=pdf_directory_path)
documents = reader.load_data()
print(f"  Ingestion Complete: Successfully parsed {len(documents)} document pages.")

# Parse text elements, generate embeddings via API, and bundle into a queryable memory index
print("  Generating dense text vector embeddings via OpenAI API...")
index = VectorStoreIndex.from_documents(documents)
print("  Vector Indexing Process complete. In-memory engine online.")


    
    # --------------------------------------------------------------------------
    # LlamaIndex Question 1: Query Engine Orchestration & Node Inspection
    # --------------------------------------------------------------------------
print("\n# LlamaIndex Question 1: Running Query Engine & Source Analysis Loops")
    
    # Initialize a query engine specifying top_k retrieval parameters
query_engine = index.as_query_engine(similarity_top_k=3)
    
questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]
    
for q_idx, query_text in enumerate(questions, start=1):
        print("\n" + "="*80)
        print(f"QUERY #{q_idx}")
        print(f"Question: {query_text}")
        print("="*80)
        
        # Dispatch query text to retrieval and synthesis pipeline
        response = query_engine.query(query_text)
        
        print(f"Answer: {response.response.strip()}\n")
        print("Retrieved Source Nodes (Top 3):")
        print("-" * 40)
        
        # Iterate over source chunks attached to the response payload
        for node_idx, source_node in enumerate(response.source_nodes, start=1):
            score = source_node.score
            score_str = f"{score:.4f}" if score is not None else "N/A"
            
            # Extract plain text content and slice clean characters
            node_text = source_node.node.get_content().strip().replace("\n", " ")
            text_snippet = node_text[:150] + "..." if len(node_text) > 150 else node_text
            
            print(f"  Node {node_idx}:")
            print(f"    Similarity Score: {score_str}")
            print(f"    Snippet (150 chars): {text_snippet}")
            print()
            
else:
    print(f"  Warning/Error: Target directory path '{pdf_directory_path}' was not found.")
    print("  Please verify your folder mapping configuration settings layout.")

    # LlamaIndex Question 1 Post-Execution Evaluation Work Block
    # --------------------------------------------------------------------------
    #
    # --- Query 1 Analysis ("What employee benefits does BrightLeaf offer?") ---
    # 1. Do the retrieved chunks look relevant to the question?
    #    Node 1 is highly relevant (Similarity Score: ~0.753) because it contains 
    #    the core foundational employee benefits overview text. However, Nodes 2 and 3 
    #    drop below 0.60 and represent macro partnership files and general company mission context.
    #
    # 2. Does the model's response sound confident and specific, or does it hedge?
    #    Note what you observe about the tone:
    #    The model's response sounds completely confident, direct, and specific. It avoids 
    #    any defensive hedging phrases like "based on the context" or "I am not sure." This is because the 
    #    exact ground-truth data was successfully passed to the context window, allowing it to list facts 
    #    declaratively (e.g., 401k match, Wellness Reimbursement Plan).
    #
    # 3. Did anything unexpected get retrieved?
    #    Yes, Node 2 ('EcoVolt Energy Partnership') was unexpectedly retrieved with 
    #    a ~0.597 score. This occurred because loose semantic keywords regarding operational collaboration 
    #    or company structural resources crossed thresholds into the top-k selection filter space.
    #
    # --- Query 2 Analysis ("What are BrightLeaf's security policies?") ---
    # 1. Do the retrieved chunks look relevant to the question?
    #    Node 1 is perfectly relevant (Similarity Score: ~0.690) because it maps 
    #    directly to the explicit network infrastructure and device authentication protocols text block.
    #
    # 2. Does the model's response sound confident and specific, or does it hedge?
    #    Note what you observe about the tone:
    #    The tone is formal, professional, and authoritative. It provides exact parameters 
    #    (such as credential rotation every 90 days and alignment with NIST guidelines) with complete certainty, 
    #    proving that RAG eliminates conversational speculation when specific data points are visible.
    #
    # 3. Did anything unexpected get retrieved?
    #    Yes, Node 2 ('Introduction / Benefits overview') and Node 3 ('EcoVolt Energy') 
    #    were unexpectedly retrieved. Because the query engine was forced to pull exactly 'similarity_top_k=3' 
    #    chunks, and the single dedicated security policy page was exhausted, the engine backfilled the remaining 
    #    slots with the next highest matching company background fragments.
    # --------------------------------------------------------------------------

# LlamaIndex Question 2: Hyperparameter Variations (Top_K Analysis)

    print("\n# LlamaIndex Question 2: Running Top_K Parametric Variations")
    
    q2_query = "What employee benefits does BrightLeaf offer?"
    top_k_variations = [1, 5]
    
    for k in top_k_variations:
        print("\n" + "="*80)
        print(f"RUNNING RETRIEVAL VARIATION: similarity_top_k={k}")
        print("="*80)
        
        query_engine_k = index.as_query_engine(similarity_top_k=k)
        response_k = query_engine_k.query(q2_query)
        
        print(f"Question: {q2_query}")
        print(f"Answer: {str(response_k).strip()}\n")
        print(f"Retrieved Source Node Scores (Total: {len(response_k.source_nodes)}):")
        
        for idx, node in enumerate(response_k.source_nodes, start=1):
            score_str = f"{node.score:.4f}" if node.score is not None else "N/A"
            print(f"  * Node {idx} Similarity Score: {score_str}")
        print()

  # --------------------------------------------------------------------------
    # LlamaIndex Question 2 Post-Execution Evaluation Work Block
    # --------------------------------------------------------------------------
    # --- LlamaIndex Q2 Reflection Commentary ---
    # Observations and Performance Evaluation:
    #
    # 1. How the response changed between the two execution configurations:
    #    - Factual Content: The core substance of the answers remained virtually identical between both runs.
    #      Both outputs listed the 401(k) retirement plan, medical insurance, wellness stipends, and the learning hub courses.
    #    - Textual Nuances: The response for similarity_top_k=5 subtly shifted its ending emphasis toward corporate values.
    #      It added the concluding sentence: "BrightLeaf emphasizes diversity, equity, and inclusion through its benefits
    #      and supports employee well-being both personally and professionally." This broader language was pulled because it
    #      had access to lower-scoring nodes (like Nodes 4 and 5 which hover way down at ~0.487 and ~0.447) that contain
    #      general company culture statements rather than explicit benefits definitions.
    #
    # 2. Is more retrieved context always better?
    #    - No, more retrieved context is definitely not always better in a production RAG system. While increasing top_k
    #      can pull fine-grained details that happen to be scattered across multiple pages, it introduces significant technical trade-offs:
    #
    #      * Prompt Distraction and Hallucination Risk: Passing weak, low-scoring fragments (like Nodes 4 and 5 which drop below a 0.50 score)
    #        floods the LLM's working window with unrelated filler words and noise. This forces the model to sift through irrelevant data,
    #        increasing processing latency and elevating the risk of semantic confusion or text hallucinations.
    #      * Cost Inflation: Every character retrieved must be passed into the API payload text. Increasing top_k from 1 to 5 dramatically
    #        inflates your input token count, causing enterprise operating expenses to shoot up over time.
    #      * The 'Lost in the Middle' Effect: Studies show that LLMs are excellent at reading text at the very beginning and very end of a prompt,
    #        but they frequently ignore or completely miss critical facts buried in the middle of long, crowded context windows.
    # --------------------------------------------------------------------------

    # LlamaIndex Question 3: Adversarial/Stress-Testing Queries
    # --------------------------------------------------------------------------
    print("\n# LlamaIndex Question 3: Executing Vague or Out-of-Bounds Queries")
    
    # Define a query designed to push structural limits or transcend dataset boundaries
    vague_or_missing_query = "How does BrightLeaf regulate zero-gravity coolant systems in space stations?"
    
    print("\n" + "="*80)
    print(f"ADVERSARIAL STRESS TEST")
    print("="*80)
    print(f"Question: {vague_or_missing_query}")
    
    query_engine_stress = index.as_query_engine(similarity_top_k=3)
    response_stress = query_engine_stress.query(vague_or_missing_query)
    
    print(f"Answer: {str(response_stress).strip()}\n")
    print("Retrieved Source Chunks:")
    print("-" * 40)
    
    for idx, node in enumerate(response_stress.source_nodes, start=1):
        score_str = f"{node.score:.4f}" if node.score is not None else "N/A"
        raw_content = node.node.get_content().strip().replace("\n", " ")
        snippet = raw_content[:150] + "..." if len(raw_content) > 150 else raw_content
        print(f"  Node {idx} [Score: {score_str}]: {snippet}\n")

  # LlamaIndex Question 3 Post-Execution Evaluation Work Block
    # --------------------------------------------------------------------------
    # --- LlamaIndex Q3 Post-Execution Reflection Commentary ---
    # Observations and Performance Evaluation:
    #
    # 1. What I Expected:
    #    I expected the pipeline to struggle significantly or completely fail to synthesize a coherent answer. This query forces a cross-document 
    #    comparison by blending details from two completely different operational domains: retirement caps ('employee_benefits.pdf') and 
    #    MFA compliance loops ('security_policy.pdf'). Because the query is split across two topics, its semantic vector embedding is inherently 
    #    'diluted.' It does not point to a single document space, which mathematically forces individual similarity scores to plummet.
    #
    # 2. What Actually Happened:
    #    The pipeline generated an astoundingly accurate response, but the underlying retrieval metrics prove it was running on thin ice. 
    #    As expected, the similarity scores dropped into a low-confidence range:
    #    - Node 1 (Security Policy): Score of ~0.391
    #    - Node 2 (Employee Benefits): Score of ~0.301
    #    - Node 3 (Financial Overview): Score of ~0.167
    #
    #    Because our configuration used 'similarity_top_k=3', both Node 1 and Node 2 were barely caught in the retrieval net and passed 
    #    into the context window. The LLM then used its internal reasoning capability to extract the discrete facts ("up to five percent" 
    #    and "rotated every 90 days") and stitch them into a clean comparison sentence. If our top_k boundary had been restricted to 1, or 
    #    if our vector store enforced a strict similarity threshold cut-off of >0.50, the system would have dropped the benefits context 
    #    entirely and failed to complete the request.
    #
    # 3. Architectural Changes to Handle This Better:
    #    To make a production RAG system natively resilient against complex, multi-hop, or comparative questions, I would implement 
    #    the following technical improvements:
    #
    #    - Query Deconstruction (Agentic Step): Place an LLM supervisor routing agent at the front gate to break complex prompts down into 
    #      independent sub-queries (e.g., Sub-Query 1: 'What are retirement matching caps?' and Sub-Query 2: 'What are MFA compliance cycles?').
    #    - Multi-Route Vector Retrieval: Execute independent vector similarity lookups for each deconstructed sub-query, pull the unique 
    #      top matches for each distinct branch, and combine the aggregated text chunks into a unified context bundle before synthesis.
    #    - Hybrid Search (BM25 + Vector): Blending keyword matching with dense vector semantics ensures that unique string entities like 
    #      'retirement matching caps' or 'multi-factor authentication' pull their respective source paragraphs instantly, bypassing 
    #      the similarity score dilution seen in a pure semantic lookup.
    # --------------------------------------------------------------------------

     # LlamaIndex Question 4: Automated Evaluation Heuristics (LLM-as-a-Judge)
    # --------------------------------------------------------------------------
    print("\n# LlamaIndex Question 4: Instantiating Programmatic Judges")
    
    # Initialize the automated evaluation structures using gpt-4o-mini
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)
    
    eval_query_engine = index.as_query_engine(similarity_top_k=3)
    
    eval_scenarios = [
        {
            "label": "Domain Match (Expected High Quality)", 
            "query": "What employee benefits does BrightLeaf offer?"
        },
        {
            "label": "Domain Mismatch (Expected Low Quality)", 
            "query": "What are the registration rules for piloting deep-sea submarines?"
        }
    ]
    
    for scenario in eval_scenarios:
        print("\n" + "="*80)
        print(f"EVALUATION SCENARIO: {scenario['label']}")
        print("="*80)
        print(f"Question: {scenario['query']}")
        
        # Run standard query string execution
        eval_response = eval_query_engine.query(scenario['query'])
        print(f"Answer: {str(eval_response).strip()}\n")
        
        # Algorithmic evaluation processing calls
        print("Calculating algorithmic metric evaluations...")
        faith_result = faithfulness_evaluator.evaluate_response(response=eval_response)
        relev_result = relevancy_evaluator.evaluate_response(response=eval_response)
        
        print(f"  >> Faithfulness Metric Score: {faith_result.score} (Passing: {faith_result.passing})")
        print(f"  >> Relevancy Metric Score:    {relev_result.score} (Passing: {relev_result.passing})")
        print()

    # --------------------------------------------------------------------------
    # LlamaIndex Question 4 Post-Execution Reflection Commentary
    # --------------------------------------------------------------------------
    # # Framework Q4 Reflection Commentary:-
    # # Metric Interpretations: A Faithfulness score of 1.0 confirms every generated claim directly mapsto reference context, whereas 0.0 marks a hallucination.
    #
    # # Relevancy tracks if the response directlyaddresses the user prompt intent, which differs from checking factual source adherence.
    #
    # # Evaluation Rationale: The "LLM-as-a-judge" framework uses semantic validation loops.
    # # This providesa flexible, meaning-aware grader that handles fluid phrasing, which rigid string matching or BLEU math cannot.
    # --------------------------------------------------------------------------


