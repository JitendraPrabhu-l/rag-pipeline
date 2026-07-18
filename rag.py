import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from loaders import load_documents

# 1. Local Knowledge Base
# Pass a file or directory (txt, md, csv, json, pdf, doc, docx, xlsx, xls, html) as
# argv[1], or drop files into ./data — otherwise a small hardcoded sample is used.
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"

if DATA_PATH.exists() and (DATA_PATH.is_file() or any(DATA_PATH.iterdir())):
    documents = load_documents(DATA_PATH)
    print(f"Loaded {len(documents)} document chunk(s) from {DATA_PATH}")
else:
    documents = [
        {
            "id": 1,
            "text": "The company expense policy states that meals are covered up to fifty dollars per day.",
            "source": "sample",
        },
        {
            "id": 2,
            "text": "Remote workers can request an ergonomic chair upgrade once every two years.",
            "source": "sample",
        },
        {
            "id": 3,
            "text": "The annual tech conference takes place in San Francisco from October 12th to 15th.",
            "source": "sample",
        },
        {
            "id": 4,
            "text": "All code production deployments require approval from at least two senior engineers.",
            "source": "sample",
        },
    ]
df = pd.DataFrame(documents)

# Minimum cosine similarity a retrieved document must clear before it's trusted as
# relevant. Below this, the query is treated as "no answer found" rather than
# returning the nearest-anyway match. Tuned empirically against short, keyword-style
# queries typed against a small, single-domain corpus like the sample above.
SIMILARITY_THRESHOLD = 0.15


# 2. Custom Tokenizer (Text Cleaning)
def tokenize(text):
    """Converts text to lowercase and splits it into discrete words."""
    return re.findall(r"\b\w+\b", text.lower())


# 3. Build Vocabulary Matrix mapping from Scratch
all_tokens = [tokenize(doc) for doc in df["text"]]
vocabulary = sorted(list(set(word for doc_words in all_tokens for word in doc_words)))
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

# 4. Compute TF-IDF Matrix via Pure NumPy & Pandas
N = len(df)
M = len(vocabulary)

# Term Frequency (TF) Matrix: shape (N, M)
TF = np.zeros((N, M))
for doc_idx, doc_words in enumerate(all_tokens):
    for word in doc_words:
        if word in word_to_idx:
            TF[doc_idx, word_to_idx[word]] += 1

# Document Frequency (DF) and Inverse Document Frequency (IDF)
df_counts = np.sum(TF > 0, axis=0)
IDF = np.log((1 + N) / (1 + df_counts)) + 1  # Smooth IDF formula

# Final TF-IDF Weight Matrix
TF_IDF_matrix = TF * IDF


# 5. Retrieval Engine (Cosine Similarity in NumPy)
def retrieve_best_document(query: str) -> dict | None:
    """Returns the best-matching {"text", "source", "score"}, or None if nothing clears SIMILARITY_THRESHOLD."""
    query_words = tokenize(query)

    # Vectorize the raw query using our built vocabulary space
    query_tf = np.zeros(M)
    for word in query_words:
        if word in word_to_idx:
            query_tf[word_to_idx[word]] += 1
    query_tfidf = query_tf * IDF

    # Calculate Cosine Similarity Vector: (A · B) / (||A|| * ||B||)
    dot_products = np.dot(TF_IDF_matrix, query_tfidf)
    matrix_norms = np.linalg.norm(TF_IDF_matrix, axis=1)
    query_norm = np.linalg.norm(query_tfidf)

    # Avoid division by zero bugs for empty queries
    if query_norm == 0 or np.all(matrix_norms == 0):
        return None

    similarities = dot_products / (matrix_norms * query_norm)
    best_match_idx = np.argmax(similarities)
    best_score = float(similarities[best_match_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return None

    return {
        "text": df.loc[best_match_idx, "text"],
        "source": df.loc[best_match_idx].get("source", "sample"),
        "score": best_score,
    }


STOPWORDS = {"what", "is", "the", "for", "a", "an", "of", "in", "to"}


# 6. Rule-Based Generative Extractor (The "No-Model" Generator)
def generate_answer(query: str, context: str) -> str:
    """Extracts the exact clause or sentence containing query keywords, or reports no answer found."""
    query_keywords = set(tokenize(query)) - STOPWORDS
    sentences = re.split(r"(?<=[.!?])\s+", context)

    best_sentence = None
    max_overlap = 0

    for sentence in sentences:
        sentence_words = set(tokenize(sentence))
        overlap = len(query_keywords.intersection(sentence_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence = sentence.strip()

    if best_sentence is None:
        return "No answer found: the retrieved document has no sentence matching the query."

    return f"Based on the system documents: {best_sentence}"


# --- Execution Loop Pipeline ---
def run_deterministic_rag(query: str):
    print(f"\n--- User Query: {query} ---")
    retrieval = retrieve_best_document(query)

    if retrieval is None:
        print("[Retrieved Context Document]: none cleared the similarity threshold.")
        print("[Deterministic Response]: No answer found: nothing in the knowledge base is relevant to this query.")
        return

    print(
        f"[Retrieved Context Document] (source={retrieval['source']}, score={retrieval['score']:.3f}): "
        f"{retrieval['text']}"
    )

    final_output = generate_answer(query, retrieval["text"])
    print(f"[Deterministic Response]: {final_output}")


# Test Cases
# run_deterministic_rag("What is the cost limit for meals?")
run_deterministic_rag(
    "Aircraft and Owner/Operator Information"
)
