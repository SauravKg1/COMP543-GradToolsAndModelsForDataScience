import re
import numpy as np
import ast

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Assignment4_kNN_20News").getOrCreate()
sc = spark.sparkContext

# -----------------------------------------------------------------------------
# Task 1: Build dictionary of top 20,000 words and document count vectors
# -----------------------------------------------------------------------------

# Load up all of the 19,997 documents in the corpus.
corpus = sc.textFile("s3://luisguzmannateras/Assignment4/20_news_same_line.txt")

# Each entry in valid_lines will be a line from the text file.
valid_lines = corpus.filter(lambda x: 'id' in x)

# Transform into a bunch of (docID, text) pairs.
key_and_text = valid_lines.map(
    lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:])
)

# Split the text in each (docID, text) pair into a list of words.
# After this, we have (docID, ["word1", "word2", "word3", ...]).
regex = re.compile('[^a-zA-Z]')
key_and_list_of_words = key_and_text.map(
    lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split())
).cache()

# Change (docID, ["word1", "word2", ...]) to ("word1", 1), ("word2", 1), ...
all_words = key_and_list_of_words.flatMap(lambda x: ((w, 1) for w in x[1]))

# Count all of the words, giving us ("word1", c1), ("word2", c2), etc.
all_counts = all_words.reduceByKey(lambda a, b: a + b)

sorted_word_counts = (
    all_counts
    .sortBy(lambda wc: (-wc[1], wc[0]))
    .take(20000)
)

# Build a local dictionary mapping: word -> dictionary index (0..19999)
dictionary = {word: idx for idx, (word, _) in enumerate(sorted_word_counts)}
vocab_size = len(dictionary)

# Broadcast dictionary for use inside RDD transformations.
broadcast_dict = sc.broadcast(dictionary)

def words_to_count_vector(words):
    """
    Convert a list of tokens for a document into a NumPy count vector
    of length = vocab_size, where entry i is the count of the ith word
    in the dictionary.
    """
    vec = np.zeros(vocab_size, dtype=np.float32)
    mapping = broadcast_dict.value
    for w in words:
        idx = mapping.get(w)
        if idx is not None:
            vec[idx] += 1.0
    return vec

# RDD whose value is a NumPy count vector for each document:
# (docID, numpy_array length 20000 with raw counts)
doc_id_and_counts = key_and_list_of_words.map(
    lambda kv: (kv[0], words_to_count_vector(kv[1]))
).cache()

def print_nonzero_counts_for_doc(doc_id):
    """
    Helper to print non-zero entries of the count vector for a given document.
    This matches the requirement in Task 1.
    """
    result = doc_id_and_counts.lookup(doc_id)
    if not result:
        print("Document not found:", doc_id)
        return
    vec = result[0]
    nz_idx = vec.nonzero()[0]
    print("Document:", doc_id)
    print("Non-zero count vector entries (index, value):")
    print(list(zip(nz_idx.tolist(), vec[nz_idx].tolist())))
    print()

# Number of documents in the corpus.
num_docs = doc_id_and_counts.count()

# Compute document frequencies for each dictionary index.
def words_to_index_set(words):
    """
    For a list of tokens, return the set of dictionary indices that appear
    at least once in the document.
    """
    mapping = broadcast_dict.value
    indices = set()
    for w in words:
        idx = mapping.get(w)
        if idx is not None:
            indices.add(idx)
    return list(indices)

# RDD of lists of indices (one list per document)
doc_index_sets = key_and_list_of_words.map(lambda kv: words_to_index_set(kv[1]))

# Compute df(i): number of documents containing word i
df_counts = doc_index_sets.flatMap(lambda idxs: ((i, 1) for i in idxs)) \
                          .reduceByKey(lambda a, b: a + b)

df_dict = dict(df_counts.collect())

# Build IDF vector as a NumPy array on the driver.
idf = np.zeros(vocab_size, dtype=np.float32)
for i in range(vocab_size):
    df = df_dict.get(i, 1)
    # Standard TF–IDF IDF definition:
    #   IDF(i) = log10(N / df(i))
    idf[i] = np.log10(float(num_docs) / float(df))

idf_broadcast = sc.broadcast(idf)

def count_to_tfidf(count_vec):
    """
    Convert a NumPy count vector into a TF–IDF vector using:
      TF(i, d)   = count(i, d) / (total number of words in d)
      IDF(i)     = log10(N / df(i))
      TF–IDF(i)  = TF(i, d) * IDF(i)
    """
    total = float(count_vec.sum())
    if total == 0.0:
        return np.zeros_like(count_vec, dtype=np.float32)
    tf = count_vec / total
    return tf * idf_broadcast.value

# RDD: (docID, TF–IDF vector)
doc_id_and_tfidf = doc_id_and_counts.mapValues(count_to_tfidf).cache()

def print_nonzero_tfidf_for_doc(doc_id):
    """
    Helper to print non-zero entries of the TF–IDF vector for a given document.
    """
    result = doc_id_and_tfidf.lookup(doc_id)
    if not result:
        print("Document not found:", doc_id)
        return
    vec = result[0]
    nz_idx = vec.nonzero()[0]
    print("Document:", doc_id)
    print("Non-zero TF–IDF vector entries (index, value):")
    print(list(zip(nz_idx.tolist(), vec[nz_idx].tolist())))
    print()

def extract_category(doc_id):
    """
    Extract newsgroup category from a document identifier of the form:
      20_newsgroups/category_name/doc_number
    """
    parts = doc_id.split('/')
    if len(parts) >= 3:
        return parts[1]
    return ""

# RDD: (docID, category, TF–IDF vector)
docid_cat_tfidf = doc_id_and_tfidf.map(
    lambda kv: (kv[0], extract_category(kv[0]), kv[1])
).cache()

def text_to_tfidf_vector(text):
    """
    Convert a raw query string into a TF–IDF vector using the
    dictionary and IDF values computed from the corpus.
    """
    tokens = regex.sub(' ', text).lower().split()
    mapping = broadcast_dict.value
    vec = np.zeros(vocab_size, dtype=np.float32)
    for w in tokens:
        idx = mapping.get(w)
        if idx is not None:
            vec[idx] += 1.0
    return count_to_tfidf(vec)

def predict_label(k, query):
    q_vec = text_to_tfidf_vector(query)

    def distance_with_label(triple):
        doc_id, category, vec = triple
        diff = vec - q_vec
        dist = float(np.sqrt(np.dot(diff, diff)))  # Euclidean distance
        return (dist, doc_id, category)

    # RDD of (dist, docID, category)
    distances = docid_cat_tfidf.map(distance_with_label)

    # Take the k closest documents.
    k_nearest = distances.takeOrdered(k, key=lambda x: x[0])

    print("k nearest neighbors for query:")
    for dist, doc_id, category in k_nearest:
        print("  distance = {:.6f}, docID = {}, category = {}".format(
            dist, doc_id, category
        ))

    # Majority vote over categories; break ties by closest neighbor.
    from collections import Counter
    counts = Counter(cat for (_, _, cat) in k_nearest)
    max_votes = max(counts.values())
    tied_categories = {cat for cat, c in counts.items() if c == max_votes}

    if len(tied_categories) == 1:
        predicted = next(iter(tied_categories))
    else:
        # Tie: pick the category of the closest neighbor among the tied ones.
        predicted = None
        for dist, doc_id, category in k_nearest:
            if category in tied_categories:
                predicted = category
                break

    print("Predicted label:", predicted)
    print()
    return predicted

if __name__ == "__main__":
    # Task 1: print non-zero count entries for the specified documents.
    docs_to_show = [
        "20_newsgroups/alt.atheism/51265",
        "20_newsgroups/talk.politics.misc/176982",
        "20_newsgroups/sci.med/59273",
    ]
    for doc_id in docs_to_show:
        print_nonzero_counts_for_doc(doc_id)

    # Task 2: print non-zero TF–IDF entries for the same documents.
    for doc_id in docs_to_show:
        print_nonzero_tfidf_for_doc(doc_id)

    # Task 3: read queries from a4_queries.txt and classify them.
    try:
        with open("a4_queries.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line is of the form: predict_label (k, 'query text...')
                # Extract the tuple "(k, 'query text...')" and parse with ast.literal_eval.
                start = line.find("(")
                end = line.rfind(")")
                args = ast.literal_eval(line[start:end + 1])
                k_val, query_text = args
                print("Query text:")
                print(query_text)
                predicted = predict_label(k_val, query_text)
                print("Predicted category:", predicted)
                print("-" * 80)
    except FileNotFoundError:
        print("a4_queries.txt not found; Task 3 classification not executed.")
