# For your reference, here is the dictionary-based LDA for use with the 
# first sub-problem.

import numpy as np
import time

# This returns a number whose probability of occurence is p.
def sample_value(p):
    """Samples a single value from a multinomial distribution."""
    return np.flatnonzero(np.random.multinomial(1, p, 1))[0]


# There are 2000 words in the corpus.
alpha = np.full(2000, .1)

# There are 100 topics.
beta = np.full(100, .1)

# This gets us the probabilty of each word happening in each of the 100 topics.
words_in_topic = np.random.dirichlet(alpha, 100)
# words_in_corpus[i] will be a dictionary that gives us the number of 
# each word in the document.
wordsInCorpus = {}
words_in_corpus = np.zeros((50, 2000), dtype=int)  # array: docs x vocab

# Generate each doc.
for doc in range(0, 50):
    # No words in this doc yet.
    words_in_doc = {}

    # Get the topic probabilities for this doc.
    topics_in_doc = np.random.dirichlet(beta)

    # Assign each of the 2000 words in this doc to a topic (aggregate).
    words_to_topic = np.random.multinomial(2000, topics_in_doc)

    # For each topic, distribute its word-count across the 2000 vocab entries.
    for topic in range(100):
        count_t = words_to_topic[topic]
        if count_t == 0:
            continue
        words_counts = np.random.multinomial(count_t, words_in_topic[topic])

        # Update dictionary version for nonzero counts
        nz = np.nonzero(words_counts)[0]
        for w in nz:
            c = int(words_counts[w])
            words_in_doc[w] = words_in_doc.get(w, 0) + c
            words_in_corpus[doc, w] += c

    # Save dictionary for this doc
    wordsInCorpus[doc] = words_in_doc

# =============================================================================
# Task 1 - Dictionary-based Operations
# Compute co-occurrences by looping over documents and word pairs.
# Store each pair once as (min(i,j), max(i,j)) to match the spec.
# =============================================================================
start = time.time()

# coOccurrences should be a map where the key is a (wordOne, wordTwo)
# pair and the value is the number of documents where both appeared.
coOccurrences_dict = {}

for doc in wordsInCorpus:
    inner = wordsInCorpus[doc]
    # Get the set of words that appear in this doc (binary presence)
    words = list(inner.keys())
    # For each pair (including diagonal)
    for i, wi in enumerate(words):
        # diagonal
        key = (wi, wi)
        coOccurrences_dict[key] = coOccurrences_dict.get(key, 0) + 1
        for wj in words[i+1:]:
            # canonicalize to single key per unordered pair
            a, b = (wi, wj) if wi <= wj else (wj, wi)
            key_ab = (a, b)
            coOccurrences_dict[key_ab] = coOccurrences_dict.get(key_ab, 0) + 1

end = time.time()
print("Task 1 (dict loops) runtime:", end - start)

# =============================================================================
# Task 2 - Using NumPy Arrays (outer product per document)
# =============================================================================
start = time.time()

coOccurrences_outer = np.zeros((2000, 2000), dtype=int)

for d in range(words_in_corpus.shape[0]):
    vec = words_in_corpus[d]
    # Clip counts to binary presence
    vec_bin = np.clip(vec, 0, 1)
    # Outer product gives 1s where both present
    coOccurrences_outer += np.outer(vec_bin, vec_bin).astype(int)

end = time.time()
print("Task 2 (NumPy outer per doc) runtime:", end - start)

# =============================================================================
# Task 3 - Using Matrix Multiplication (single multiply)
# =============================================================================
start = time.time()

B = np.clip(words_in_corpus, 0, 1)   # binary presence matrix (50 x 2000)
coOccurrences_mm = B.T.dot(B)        # (2000 x 2000)

end = time.time()
print("Task 3 (matrix multiply) runtime:", end - start)
