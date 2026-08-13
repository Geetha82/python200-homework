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

# --- RAG Concepts ---

# Concepts Question 1
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

# Concepts Q1: Evaluation of Augmentation Architectures
# 1. Context Injection:
# - Pros: Zero training cost, highly accurate for provided text, instant updates.
# - Cons: Bound by context window limits; linear cost growth with input token volume.

# 2. Fine-Tuning:
# - Pros: Optimizes formatting, style, tone, and deep domain-specific vocabulary rules.
# - Cons: Computationally expensive; slow iteration cycle; prone to factual hallucinations.

# 3. Retrieval-Augmented Generation (RAG):
# - Pros: Scalable to massive external enterprise corpuses; verifiable with audit trails/citations.
# - Cons: Relies heavily on retrieval accuracy; higher orchestration complexity.

# Scenario Recommendations Log:
# Scenario A -> Best Approach: RAG (Handles hundreds of dynamically changing quarterly PDFs with strict traceability).
# Scenario B -> Best Approach: Fine-Tuning (Alters behavioral stylistic weight parameters using 3,000 target examples).
# Scenario C -> Best Approach: Prompt Engineering (Direct context inj

# Concepts Question 2
print("\n# Concepts Q2: Analysis of Confident Hallucinations")
hallucination_analysis = """
- Harm Threshold: "I am not sure" preserves human vigilance; confident errors systematically bypass human skepticism.
- Clinical/Real-World Example: A medical diagnostic bot misinterpreting a critical symptom as benign, delaying life-saving medical care.
- Tone vs. Trust: Fluent, highly authoritative phrasing mirrors genuine expertise, exploiting linguistic biases to mask baseline statistical uncertainty.
"""
print(hallucination_analysis.strip())

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

# Concepts Question 3
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

# Concepts Q3: The steps below make up a complete RAG pipeline, but they are out of order.
# Original out-of-order list from the prompt:
# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]

# Reordered steps with a one-sentence description for each:
# 1. Extract text from source documents - This step reads the raw text contents out of your source files.
# 2. Split text into chunks - This breaks long continuous text down into smaller, manageable portions.
# 3. Convert text chunks into embeddings - This turns each text piece into a list of numbers representing its meaning.
# 4. Receive the user's query - This collects the natural language question or prompt typed by the user.
# 5. Embed the user's query - This converts the user's question into a list of numbers using the same embedding model.
# 6. Retrieve the most relevant chunks - This compares the query numbers to the text numbers to find the closest matches.
# 7. Inject retrieved chunks into the prompt - This pastes the best matching text blocks into the final prompt instructions.
# 8. Generate a response from the LLM - This lets the model read the complete context and write out the final factual answer.



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
shared_documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


# Keyword Question 1
# Selected Document: hours.txt
# Explanation: The keywords extracted from the query are 'hours' and 'weekends'. The document 'hours.txt' contains an exact match for both tokens, yielding the highest overlap score of 2, which makes it the selected document.

print("\n=== Running: Keyword RAG Q1 ===")
query_1 = "What are your hours on weekends?"
results_1 = simple_keyword_retrieval(query_1, shared_documents, verbose=True)
print(f"Final Selected Document: {results_1[0][0]}")


# Keyword Question 1
# Selected Document: hours.txt
# Explanation:
#  Based on the terminal output, the filtered query tokens are ['hours', 'weekends', 'what']. 
# The document 'hours.txt' was selected because it is the only document that features a matching keyword token ('weekends'), 
# giving it a top overlap score of 1 while all other files registered 0.

# Keyword Question 2
print("\n# Keyword Question 2: Running a No-Overlap Search Vector Query")

q2_query = "Do you have anything without caffeine?"
# Uses the exact same document dictionary corpus from Keyword Q1

# Execute keyword search using our shared documents corpus dictionary
q2_results = simple_keyword_retrieval(q2_query, shared_documents, verbose=True)

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


# Keyword Question 3
print("\n# Keyword Question 3: Running a Synonym Gap Search Query")
query_3 = "How do I sign up for rewards?"
# Execute keyword search using our shared documents corpus 
dictionaryresults_3 = simple_keyword_retrieval(query_3, shared_documents, verbose=True)
print(f"Final Selected Document: {dictionaryresults_3[0][0]}")

# Keyword Question 3 Explanation Comment Block:
# ------------------------------------------------------------------------------
# Explanation:
# 1. Prediction: It will return 'None found' or fail to find the correct document.

# 2. Reasoning: The user is looking for how to get 'rewards' or 'sign up'. 
# The file'loyalty.txt' explains how to do exactly this, but it uses the words 'loyalty program'and 'join' instead. 
# Since the exact words do not overlap, keyword search cannot match them.

# 3. Was the prediction correct? Yes. 
# The keyword system returned 'None found' with an overlapscore of 0 across all files 
# because it cannot traverse synonym gaps or conceptual mappings.


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

# --- LlamaIndex Pipelines & Evaluation ---
print("\n=== Running: LlamaIndex Pipelines & Evaluation ===")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

target_pdf_dir = "./brightleaf_pdfs"

# Stop the script with a clear message if the directory path is wrong
if not os.path.exists(target_pdf_dir):
    raise FileNotFoundError(f"Error: Could not find the folder at: {target_pdf_dir}")

# Set up the OpenAI embedding model framework choice
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 1. Read files out of the directory
print("Reading files from folder...")
loaded_documents = SimpleDirectoryReader(input_dir=target_pdf_dir).load_data()
print(f"Found and loaded {len(loaded_documents)} pages.")

# 2. Convert text to numbers and build search table
print("Building index tables (Sending data to OpenAI)...")
solar_index = VectorStoreIndex.from_documents(loaded_documents)
print("Index built successfully.")

# 3. Create the query tool configured to fetch the top 3 best matching blocks
rag_query_engine = solar_index.as_query_engine(similarity_top_k=3)
print("Query engine ready.\n")


# LlamaIndex Question 1

print("\n===== LlamaIndex Question 1===== \n")

# --- Run Question 1 ---
question_1 = "What employee benefits does BrightLeaf offer?"
print(f"--------------------------------------------------")
print(f"Question: {question_1}")
print(f"--------------------------------------------------")

response_obj_1 = rag_query_engine.query(question_1)
print(f"Answer: {response_obj_1.response.strip()}\n")

print("Retrieved Source Text Blocks:")
for node_idx, source_node in enumerate(response_obj_1.source_nodes, start=1):
    node_score = source_node.score if source_node.score is not None else 0.0
    raw_text = source_node.node.get_content().strip().replace('\n', ' ')
    short_preview = raw_text[:150]
    
    print(f"  [Block {node_idx}] Matching Score: {node_score:.4f}")
    print(f"  [Block {node_idx}] Preview: \"{short_preview}...\"")
    print()

# --- Run Question 2 ---
question_2 = "What are BrightLeaf's security policies?"
print(f"--------------------------------------------------")
print(f"Question: {question_2}")
print(f"--------------------------------------------------")

response_obj_2 = rag_query_engine.query(question_2)
print(f"Answer: {response_obj_2.response.strip()}\n")

print("Retrieved Source Text Blocks:")
for node_idx, source_node in enumerate(response_obj_2.source_nodes, start=1):
    node_score = source_node.score if source_node.score is not None else 0.0
    raw_text = source_node.node.get_content().strip().replace('\n', ' ')
    short_preview = raw_text[:150]
    
    print(f"  [Block {node_idx}] Matching Score: {node_score:.4f}")
    print(f"  [Block {node_idx}] Preview: \"{short_preview}...\"")
    print()

# LlamaIndex Question 1 Analysis & Explanations (Based on Actual Results)

# Query 1 Analysis: "What employee benefits does BrightLeaf offer?"
# - Do the retrieved chunks look relevant?
#   Only Block 1 is truly relevant because it captures the introduction to the benefits program. 
#   Blocks 2 and 3 are irrelevant to employee benefits; they cover a microgrid partnership and general history.
# - Does the model's response sound confident and specific, or does it hedge?
#   The model's response sounds incredibly confident, precise, and highly specific. It directly lists out health 
#   care options, retirement 401(k) matches, and professional development stipends without any hedging phrases.
# - Did anything unexpected get retrieved?
#   Yes, retrieving Block 2 (EcoVolt Energy microgrid partnership) and Block 3 (general mission overview) was unexpected, 
#   since neither piece of text addresses employee compensation perks or wellness programs.
#
# Query 2 Analysis: "What are BrightLeaf's security policies?"
# - Do the retrieved chunks look relevant?
#   Only Block 1 is directly relevant since it contains the actual network and data security architecture rules. 
#   Blocks 2 and 3 are completely irrelevant to data security policies.
# - Does the model's response sound confident and specific, or does it hedge?
#   The model's response is highly confident and comprehensive. It confidently lays out detailed technical rules 
#   like layered network defenses, MFA requirements, credential rotation, and compliance with NIST/ISO guidelines 
#   without any hesitation.
# - Did anything unexpected get retrieved?
#   Yes, Block 2 (the employee benefits introduction text) and Block 3 (the EcoVolt Energy microgrid partnership) 
#   were completely unexpected noise chunks. They show up because a small document repository can allow slightly matching 
#   organizational keywords to slip through into the top-k results.

# --- LlamaIndex Question 2 ---

print("\n===== LlamaIndex Question 2===== \n")

# Run 1: Re-running with similarity_top_k=1
print(f"\n[Run 1] Re-running question with similarity_top_k=1...")
rag_query_engine_k1 = solar_index.as_query_engine(similarity_top_k=1)
response_k1 = rag_query_engine_k1.query(question_1)

print(f"Answer (top_k=1):\n{response_k1.response.strip()}\n")
print("Source Node Scores (top_k=1):")
for node_idx, source_node in enumerate(response_k1.source_nodes, start=1):
    node_score = source_node.score if source_node.score is not None else 0.0
    print(f"  Node {node_idx} Score: {node_score:.4f}")

# Run 2: Re-running with similarity_top_k=5
print(f"\n[Run 2] Re-running question with similarity_top_k=5...")
rag_query_engine_k5 = solar_index.as_query_engine(similarity_top_k=5)
response_k5 = rag_query_engine_k5.query(question_1)

print(f"Answer (top_k=5):\n{response_k5.response.strip()}\n")
print("Source Node Scores (top_k=5):")
for node_idx, source_node in enumerate(response_k5.source_nodes, start=1):
    node_score = source_node.score if source_node.score is not None else 0.0
    print(f"  Node {node_idx} Score: {node_score:.4f}")

# LlamaIndex Question 2 Analysis & Explanations (Based on Actual Results)

# How the response changed:
# Looking closely at the output, the core answer barely changed at all! Both top_k=1 and top_k=5 
# successfully listed the exact same core benefits (medical insurance, wellness reimbursements, 401k match, 
# parental leave, etc.). This happened because Node 1 (score 0.7475) already contained the complete, correct 
# text needed to answer the question. The model didn't need any extra details from the other chunks.

# Is more retrieved context always better?
# No, more context is definitely not always better. Here is what the experiment teaches us:
# 1. Diminishing Returns & Noise: In the top_k=5 run, the matching scores dropped rapidly from 0.7476 down to 
#    0.4365. These extra chunks were mostly irrelevant "noise" (like the EcoVolt partnership).
# 2. Increased Resource Cost: Pulling 5 chunks means we are sending significantly more text tokens to the OpenAI 
#    API. This makes each query more expensive and can increase processing time (latency).
# 3. Risk of Distraction: If the extra text contains confusing or conflicting information, a model might 
#    accidentally focus on the wrong information or dilute the crisp correctness of the original answer. More 
#    context is only useful if the information is spread across multiple different files.


print("\n===== LlamaIndex Question 3===== \n")
question_3 = "What are the details of BrightLeaf's stock option plans and company financial performance metrics?"
print(f"Question: {question_3}")
print(f"--------------------------------------------------")

response_obj_3 = rag_query_engine.query(question_3)
print(f"Answer:\n{response_obj_3.response.strip()}\n")

print("Retrieved Source Text Blocks:")
for node_idx, source_node in enumerate(response_obj_3.source_nodes, start=1):
    node_score = source_node.score if source_node.score is not None else 0.0
    raw_text = source_node.node.get_content().strip().replace('\n', ' ')
    short_preview = raw_text[:150]
    
    print(f"  [Block {node_idx}] Matching Score: {node_score:.4f}")
    print(f"  [Block {node_idx}] Preview: \"{short_preview}...\"")
    print()

# LlamaIndex Question 3 Analysis & Explanation

# What was expected:
# We expected the pipeline to struggle because stock option details are completely absent from the documents, 
# meaning the model should ideally say it does not know or cannot find the stock option rules.

# What actually happened:
# The model correctly realized that stock option plans were not mentioned in the text. However, because one of the 
# documents (Block 3, score 0.5740) actually contained a summary of BrightLeaf Solar's financial performance 
# metrics from 2021 through 2025, the semantic search successfully pulled that chunk! The model combined these two 
# facts perfectly: it stated that stock plans were missing, but went ahead and listed the exact revenue figures 
# (\$2.8M to \$7.1M) and net profit metrics (\$0.3M to \$1.3M) found in the retrieved context block.

# What to change about the system to handle this kind of query better:
# 1. Similarity Score Threshold: Implement a minimum score cutoff (e.g., discard nodes with scores below 0.65). 
#    This would filter out weak matches like Block 2 (0.5875) and block out noise before it reaches the prompt.
# 2. Query Rewriting / Sub-Question Query Engine: The query blends two distinct questions (stock plans AND financial metrics). 
#    Using a router or splitting the question into two sub-queries would allow the system to evaluate each piece 
#    separately, logging a clear "Not Found" for the stock options without cluttering or stalling the retrieval loop.

# --- LlamaIndex Question 4 ---
print("\n===== LlamaIndex Question 4===== \n")

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

faithfulness_checker = FaithfulnessEvaluator(llm=Settings.llm)
relevancy_checker = RelevancyEvaluator(llm=Settings.llm)

# 1. Evaluate high-quality response
print("\n[Run 1] Evaluating High-Quality Response (Employee Benefits Query)...")
faith_result_1 = faithfulness_checker.evaluate_response(response=response_obj_1)
rel_result_1 = relevancy_checker.evaluate_response(query=question_1, response=response_obj_1)

print(f"Query: '{question_1}'")
print(f"  Faithfulness: Score = {faith_result_1.score} | Passing = {faith_result_1.passing}")
print(f"  Relevancy: Score = {rel_result_1.score} | Passing = {rel_result_1.passing}")

# 2. Evaluate low-quality response
unrelated_query = "What is the secret recipe for baking chocolate chip cookies?"
print(f"\n--------------------------------------------------")
print(f"Question: {unrelated_query}")
print("--------------------------------------------------")

response_obj_unrelated = rag_query_engine.query(unrelated_query)
print(f"Answer:\n{response_obj_unrelated.response.strip()}\n")

print("[Run 2] Evaluating Low-Quality Response (Cookie Query)...")
faith_result_unrelated = faithfulness_checker.evaluate_response(response=response_obj_unrelated)
rel_result_unrelated = relevancy_checker.evaluate_response(query=unrelated_query, response=response_obj_unrelated)

print(f"Query: '{unrelated_query}'")
print(f"  Faithfulness: Score = {faith_result_unrelated.score} | Passing = {faith_result_unrelated.passing}")
print(f"  Relevancy: Score = {rel_result_unrelated.score} | Passing = {rel_result_unrelated.passing}")

# LlamaIndex Question 4 Analysis & Explanations (Based on Terminal Results)

# 1. What does a faithfulness score of 1.0 mean?
#    A faithfulness score of 1.0 means that the generated answer is completely grounded in and supported by 
#    the retrieved text chunks. Every factual claim the model made can be traced back directly to the source text, 
#    proving there are no external hallucinations or invented information.

# 2. What would a score of 0.0 indicate?
#    A faithfulness score of 0.0 indicates that none of the claims made in the model's generated answer are 
#    supported by the retrieved source context chunks. The model completely hallucinated the response by drawing 
#    from its own pre-trained internal memory rather than relying on the specific context it was given.

# 3. What does a relevancy score measure, and how is it different from faithfulness?
#    Relevancy measures how well the generated answer directly addresses the user's initial question, regardless 
#    of whether that answer is true or supported by a file. 
#    - Faithfulness checks: "Is the answer strictly based on the retrieved documents?"
#    - Relevancy checks: "Does the answer actually satisfy what the user asked?"
#    An answer can be perfectly faithful but irrelevant (e.g., answering a question about security with an unhallucinated 
#    paragraph about employee benefits), or it can be highly relevant but completely unfaithful (like the cookie example).

# 4. Did the scores change between your two queries? If so, why do you think that happened?
#    Yes, the scores dropped dramatically from 1.0/1.0 down to 0.0/0.0. This happened because the employee benefits query 
#    had direct matching data in the solar PDFs, leading to a perfectly grounded and correct answer. 
#    However, when asked about chocolate chip cookies, the document text contains zero information about baking. 
#    Instead of admitting it didn't know, the model ignored the retrieved solar context and hallucinated a full cookie 
#    recipe. The judge LLM correctly caught this: it flagged that the answer had zero grounding in the files (Faithfulness = 0.0) 
#    and that the retrieved solar documents were completely useless for answering a baking query (Relevancy = 0.0).

# 5. What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
#    The "LLM-as-a-judge" approach uses an intelligent model (like gpt-4o-mini) to programmatically review the query, 
#    retrieved context, and answer to determine passing scores. It is used instead of a simple accuracy metric (like exact string 
#    matching or character overlap metrics) because language can express the exact same correct answer in thousands of 
#    different sentence structures. Traditional keywords or similarity scores can easily miss a correct answer that uses 
#    synonyms, whereas a judge LLM understands deep semantic meaning, logic, and context grounding.
