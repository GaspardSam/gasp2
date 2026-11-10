"""
Index-based search engine with a tf-idf matrix.

Inputs (and ONLY these two):
    1. a query string (may contain multiple words)
    2. a folder path containing the .txt documents of the corpus

Output:
    the documents ranked by relevance, each displayed with its
    first 200 characters.

Usage:
    python search_engine.py
    python search_engine.py "my query" /path/to/corpus
"""

import math
import os
import re
import sys
from collections import Counter, defaultdict


class SearchEngine:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.documents = {}              # doc_name -> raw text
        self.inverted_index = defaultdict(dict)  # term -> {doc_name: tf-idf weight}
        self.tfidf_matrix = {}           # doc_name -> {term: tf-idf weight} (sparse rows)
        self.doc_norms = {}              # doc_name -> L2 norm of its tf-idf vector
        self.idf = {}                    # term -> inverse document frequency

        self._load_documents()
        self._build_index()

    # ------------------------------------------------------------------
    # Step 1: read the corpus
    # ------------------------------------------------------------------
    def _load_documents(self):
        """Read every .txt file of the folder into memory."""
        for file_name in sorted(os.listdir(self.folder_path)):
            full_path = os.path.join(self.folder_path, file_name)
            if os.path.isfile(full_path) and file_name.lower().endswith(".txt"):
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    self.documents[file_name] = f.read()

        if not self.documents:
            raise ValueError(f"No .txt documents found in '{self.folder_path}'")

    # ------------------------------------------------------------------
    # Step 2: text preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def tokenize(text):
        """Lowercase the text and split it into alphanumeric words."""
        return re.findall(r"\b\w+\b", text.lower())

    # ------------------------------------------------------------------
    # Step 3: build the tf-idf matrix and the inverted index
    # ------------------------------------------------------------------
    def _build_index(self):
        n_docs = len(self.documents)

        # Term frequencies (raw counts) for each document
        term_counts = {name: Counter(self.tokenize(text))
                       for name, text in self.documents.items()}

        # Document frequency: in how many documents does each term appear?
        doc_freq = Counter()
        for counts in term_counts.values():
            doc_freq.update(counts.keys())

        # Smoothed IDF: log((1 + N) / (1 + df)) + 1  (never zero, never negative)
        self.idf = {term: math.log((1 + n_docs) / (1 + df)) + 1
                    for term, df in doc_freq.items()}

        # tf-idf weights: tf = count / total number of words in the document
        for name, counts in term_counts.items():
            total_terms = sum(counts.values()) or 1
            row = {}
            for term, count in counts.items():
                weight = (count / total_terms) * self.idf[term]
                row[term] = weight
                # Inverted index: term -> documents containing it (with weight)
                self.inverted_index[term][name] = weight
            self.tfidf_matrix[name] = row
            self.doc_norms[name] = math.sqrt(sum(w * w for w in row.values()))

    # ------------------------------------------------------------------
    # Step 4: query processing and ranking
    # ------------------------------------------------------------------
    def search(self, query):
        """Return a list of (doc_name, score) sorted by decreasing cosine similarity."""
        query_tokens = [t for t in self.tokenize(query) if t in self.idf]
        if not query_tokens:
            return []  # none of the query words exists in the corpus

        # tf-idf vector of the query (same weighting scheme as documents)
        q_counts = Counter(query_tokens)
        q_total = sum(q_counts.values())
        q_vector = {t: (c / q_total) * self.idf[t] for t, c in q_counts.items()}
        q_norm = math.sqrt(sum(w * w for w in q_vector.values()))

        # Use the inverted index: only documents containing at least one
        # query term are visited (no need to scan the whole corpus).
        dot_products = defaultdict(float)
        for term, q_weight in q_vector.items():
            for doc_name, d_weight in self.inverted_index[term].items():
                dot_products[doc_name] += q_weight * d_weight

        # Cosine similarity = dot product / (||q|| * ||d||)
        scores = []
        for doc_name, dot in dot_products.items():
            denom = q_norm * self.doc_norms[doc_name]
            if denom > 0:
                scores.append((doc_name, dot / denom))

        # Rank: highest score first (ties broken alphabetically for stability)
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores

    # ------------------------------------------------------------------
    # Step 5: display results
    # ------------------------------------------------------------------
    def display_results(self, query):
        """Print the ranked results with the first 200 characters of each document."""
        results = self.search(query)
        if not results:
            print("No document matches your query.")
            return

        print(f"\n{len(results)} result(s) for the query: '{query}'\n")
        for rank, (doc_name, score) in enumerate(results, start=1):
            # Replace line breaks by spaces so the preview stays on one line
            preview = self.documents[doc_name][:200].replace("\n", " ")
            print(f"#{rank} - {doc_name} (score: {score:.4f})")
            print(f"    {preview}")
            print("-" * 60)


def main():
    # The two allowed inputs: (1) query string, (2) folder path
    if len(sys.argv) >= 3:
        query, folder = sys.argv[1], sys.argv[2]
    else:
        query = input("Enter your query: ")
        folder = input("Enter the folder path of the corpus: ")

    engine = SearchEngine(folder)
    engine.display_results(query)


if __name__ == "__main__":
    main()
