import re
import math
from operator import add

from pyspark import SparkConf, SparkContext

VOCAB_SIZE    = 20000
MAX_ITERS     = 200        # max iterations of gradient descent
LEARNING_RATE = 0.003      # initial learning rate (η), adapted by bold driver
LAMBDA_REG    = 0.001      # L2 regularization coefficient (λ)
TOL           = 1e-4       # stopping threshold on |Δ(negative LLH)|

TRAIN_PATH = "s3://luisguzmannateras/Assignment5/TrainingDataOneLinePerDoc.txt"
TEST_PATH  = "s3://luisguzmannateras/Assignment5/TestingDataOneLinePerDoc.txt"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "as", "by", "for", "on",
    "with", "from", "that", "this", "it", "is", "are", "be", "was", "were",
    "s", "at", "but", "if", "into", "than", "then"
}

def parse_doc(line):
    id_match = re.search(r'<doc\s+id=\"([^\"]+)\"', line)
    docid = id_match.group(1) if id_match else ""

    content_match = re.search(r'>\s*(.*)\s*</doc>', line)
    text = content_match.group(1) if content_match else ""
    return (docid, text)


def tokenize(text):
    """
    Simple tokenizer: lowercase and keep alphabetic tokens.
    """
    return re.findall(r"[a-zA-Z]+", text.lower())


def is_aus_case(docid):
    """
    Label: 1 if Australian court case (docid starts with 'AU'), else 0.
    """
    return 1 if docid.startswith("AU") else 0


def safe_sigmoid(z):
    """
    Numerically stable sigmoid.
    """
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def safe_log1pexp(z):
    """
    Numerically stable log(1 + exp(z)).
    """
    if z >= 0:
        ez = math.exp(-z)
        return math.log(1.0 + ez) + z
    else:
        ez = math.exp(z)
        return math.log(1.0 + ez)


def main():
    conf = SparkConf().setAppName("COMP330_543_Assignment5_LogisticRegression")
    sc = SparkContext(conf=conf)

    training_raw = sc.textFile(TRAIN_PATH)
    testing_raw  = sc.textFile(TEST_PATH)

    train_parsed = training_raw.map(parse_doc)
    test_parsed  = testing_raw.map(parse_doc)

    num_train_docs = train_parsed.count()
    num_test_docs  = test_parsed.count()
    N = float(num_train_docs)

    train_tokens = train_parsed.flatMap(lambda x: tokenize(x[1]))  # x=(docid, text)

    word_counts = train_tokens.map(lambda w: (w, 1)).reduceByKey(add)

    vocab_rdd = (
        word_counts
        .sortBy(lambda wc: (-wc[1], wc[0]))   # descending by count, then word
        .zipWithIndex()
        .filter(lambda wc_idx: wc_idx[1] < VOCAB_SIZE)
        .map(lambda wc_idx: (wc_idx[0][0], wc_idx[1]))  # (word, rank)
    )

    vocab = vocab_rdd.collect()
    word2idx = {w: idx for (w, idx) in vocab}

    for target in ["proceeding", "orders", "formula", "tribunal"]:
        rank = word2idx.get(target, -1)
        print(f"Rank of '{target}': {rank}")

    index_to_word = [None] * VOCAB_SIZE
    for w, idx in word2idx.items():
        if 0 <= idx < VOCAB_SIZE:
            index_to_word[idx] = w

    b_word2idx = sc.broadcast(word2idx)

    def doc_to_tf_train(record):
        """
        record: (docid, text)
        returns: (label, tf_dict)
        """
        docid, text = record
        label = is_aus_case(docid)
        tokens = tokenize(text)
        vocab_map = b_word2idx.value
        tf = {}
        for t in tokens:
            idx = vocab_map.get(t)
            if idx is not None:
                tf[idx] = tf.get(idx, 0.0) + 1.0
        return (label, tf)

    train_tf = train_parsed.map(doc_to_tf_train).cache()

    df_rdd = train_tf.flatMap(
        lambda lt: [(idx, 1) for idx in lt[1].keys()]
    ).reduceByKey(add)
    df_dict = dict(df_rdd.collect())

    idf = [0.0] * VOCAB_SIZE
    for idx in range(VOCAB_SIZE):
        df = df_dict.get(idx, 0.0)
        if df > 0.0:
            idf[idx] = math.log(N / df)
        else:
            idf[idx] = 0.0

    b_idf = sc.broadcast(idf)

    def tf_to_tfidf_train(lt):
        """
        lt: (label, tf_dict)
        returns: (label, tfidf_dict)
        """
        label, tf = lt
        idf_vec = b_idf.value
        tfidf = {}
        for idx, cnt in tf.items():
            val = cnt * idf_vec[idx]
            if val != 0.0:
                tfidf[idx] = val
        return (label, tfidf)

    train_tfidf = train_tf.map(tf_to_tfidf_train).cache()

    sum_x_rdd = train_tfidf.flatMap(
        lambda lt: [(idx, val) for idx, val in lt[1].items()]
    ).reduceByKey(add)

    sum_x2_rdd = train_tfidf.flatMap(
        lambda lt: [(idx, val * val) for idx, val in lt[1].items()]
    ).reduceByKey(add)

    sum_x_dict = dict(sum_x_rdd.collect())
    sum_x2_dict = dict(sum_x2_rdd.collect())

    mean = [0.0] * VOCAB_SIZE
    std  = [0.0] * VOCAB_SIZE

    for idx in range(VOCAB_SIZE):
        sx  = sum_x_dict.get(idx, 0.0)
        sx2 = sum_x2_dict.get(idx, 0.0)
        mu = sx / N
        ex2 = sx2 / N
        var = ex2 - mu * mu
        if var < 0.0:
            var = 0.0
        mean[idx] = mu
        if var > 0.0:
            std_val = math.sqrt(var)
            if std_val < 1e-12:
                std_val = 0.0
            std[idx] = std_val
        else:
            std[idx] = 0.0   # zero variance -> we will normalize to 0

    b_mean = sc.broadcast(mean)
    b_std  = sc.broadcast(std)

    def scale_features_train(lt):
        """
        lt: (label, tfidf_dict)
        returns: (label, normalized_tfidf_dict)
        (x - mean) / std, with zero-variance dims set to 0.
        """
        label, tfidf = lt
        mean_vec = b_mean.value
        std_vec  = b_std.value
        scaled = {}
        for idx, val in tfidf.items():
            sigma = std_vec[idx]
            if sigma > 0.0:
                z = (val - mean_vec[idx]) / sigma
                if z != 0.0:
                    scaled[idx] = z
            # if sigma == 0: normalized value is 0, skip
        return (label, scaled)

    train_data = train_tfidf.map(scale_features_train).cache()

    train_tfidf.unpersist()
    train_tf.unpersist()

    weights = [0.0] * VOCAB_SIZE
    prev_nll = None
    alpha = LEARNING_RATE  # bold driver learning rate

    for it in range(MAX_ITERS):
        b_w = sc.broadcast(weights)

        def seq_op(acc, record):
            grad_dict, nll_sum = acc
            label, feats = record

            w = b_w.value
            dot = 0.0
            for idx, val in feats.items():
                dot += w[idx] * val

            nll_i = safe_log1pexp(dot) - label * dot

            p = safe_sigmoid(dot)
            coeff = p - label  # (p - y)

            for idx, val in feats.items():
                grad_dict[idx] = grad_dict.get(idx, 0.0) + coeff * val

            return (grad_dict, nll_sum + nll_i)

        def comb_op(acc1, acc2):
            grad1, nll1 = acc1
            grad2, nll2 = acc2
            for idx, val in grad2.items():
                grad1[idx] = grad1.get(idx, 0.0) + val
            return (grad1, nll1 + nll2)

        init_acc = ({}, 0.0)
        grad_dict, nll_data = train_data.aggregate(init_acc, seq_op, comb_op)

        # average over number of documents
        nll_data /= N
        for idx in list(grad_dict.keys()):
            grad_dict[idx] /= N

        # add regularization to gradient and NLL
        reg_term = 0.0
        for j in range(VOCAB_SIZE):
            wj = weights[j]
            reg_term += wj * wj
            grad_dict[j] = grad_dict.get(j, 0.0) + LAMBDA_REG * wj

        nll = nll_data + 0.5 * LAMBDA_REG * reg_term  # negative LLH we minimize

        # Bold driver + stopping criterion based on change in negative LLH
        if prev_nll is not None:
            # adjust learning rate based on improvement
            if nll < prev_nll:
                alpha *= 1.05
            else:
                alpha *= 0.5

            change = abs(prev_nll - nll)
            if change < TOL:
                prev_nll = nll
                break

        prev_nll = nll

        grad = [0.0] * VOCAB_SIZE
        for idx, val in grad_dict.items():
            if 0 <= idx < VOCAB_SIZE:
                grad[idx] = val

        # gradient descent update
        for j in range(VOCAB_SIZE):
            weights[j] -= alpha * grad[j]

    # Top 20 words by largest positive coefficient, excluding stopwords
    idx_weight_pairs = list(enumerate(weights))
    idx_weight_pairs.sort(key=lambda x: -x[1])

    top_nonstop = []
    for idx, wval in idx_weight_pairs:
        if len(top_nonstop) >= 20:
            break
        if not (0 <= idx < len(index_to_word)):
            continue
        word = index_to_word[idx]
        if word is None:
            continue
        if word in STOPWORDS:
            continue
        top_nonstop.append((word, idx, wval))

    print("Top 20 words with largest positive regression coefficients (excluding stopwords):")
    for word, idx, wval in top_nonstop:
        print(f"  word={word}, index={idx}, weight={wval:.6f}")
    
    def doc_to_tf_test(record):
        """
        record: (docid, text)
        returns: (docid, label, tf_dict, text)
        """
        docid, text = record
        label = is_aus_case(docid)
        tokens = tokenize(text)
        vocab_map = b_word2idx.value
        tf = {}
        for t in tokens:
            idx = vocab_map.get(t)
            if idx is not None:
                tf[idx] = tf.get(idx, 0.0) + 1.0
        return (docid, label, tf, text)

    test_tf = test_parsed.map(doc_to_tf_test)

    def tf_to_tfidf_test(record):
        """
        record: (docid, label, tf_dict, text)
        returns: (docid, label, tfidf_dict, text)
        """
        docid, label, tf, text = record
        idf_vec = b_idf.value
        tfidf = {}
        for idx, cnt in tf.items():
            val = cnt * idf_vec[idx]
            if val != 0.0:
                tfidf[idx] = val
        return (docid, label, tfidf, text)

    test_tfidf = test_tf.map(tf_to_tfidf_test)

    def scale_features_test(record):
        """
        record: (docid, label, tfidf_dict, text)
        returns: (docid, label, normalized_tfidf_dict, text)
        """
        docid, label, tfidf, text = record
        mean_vec = b_mean.value
        std_vec  = b_std.value
        scaled = {}
        for idx, val in tfidf.items():
            sigma = std_vec[idx]
            if sigma > 0.0:
                z = (val - mean_vec[idx]) / sigma
                if z != 0.0:
                    scaled[idx] = z
        return (docid, label, scaled, text)

    test_data = test_tfidf.map(scale_features_test).cache()

    b_final_w = sc.broadcast(weights)

    def score_record(record):
        """
        record: (docid, label, scaled_features, text)
        returns: (docid, label, score=z=r·x, text)
        """
        docid, label, feats, text = record
        w = b_final_w.value
        dot = 0.0
        for idx, val in feats.items():
            dot += w[idx] * val
        return (docid, label, dot, text)

    test_scored = test_data.map(score_record).cache()

    scores_labels = test_scored.map(lambda x: (x[1], x[2])).collect()

    thresholds = [i / 100.0 for i in range(5, 96, 5)]  # 0.05 .. 0.95 step 0.05

    best_f1 = -1.0
    best_thr = 0.5
    best_metrics = (0, 0, 0, 0, 0.0, 0.0, 0.0)

    for thr in thresholds:
        tp = fp = tn = fn = 0
        for (y, z) in scores_labels:
            p = safe_sigmoid(z)
            pred = 1 if p >= thr else 0
            if y == 1 and pred == 1:
                tp += 1
            elif y == 0 and pred == 1:
                fp += 1
            elif y == 0 and pred == 0:
                tn += 1
            else:
                fn += 1

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / float(total) if total > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = (tp, fp, tn, fn, precision, recall, accuracy)

    best_tp, best_fp, best_tn, best_fn, best_prec, best_rec, best_acc = best_metrics

    # Convert probability threshold to z-threshold for reporting: z = log(p/(1-p))
    if 0.0 < best_thr < 1.0:
        best_z_thr = math.log(best_thr / (1.0 - best_thr))
    else:
        best_z_thr = None

    print(f"TP: {best_tp}")
    print(f"FP: {best_fp}")
    print(f"TN: {best_tn}")
    print(f"FN: {best_fn}")
    print(f"Precision: {best_prec:.6f}")
    print(f"Recall: {best_rec:.6f}")
    print(f"F1 score: {best_f1:.6f}")
    print(f"Accuracy: {best_acc:.6f}")

    # False positives at best threshold: Wikipedia (label=0) predicted as AU case (pred=1)
    def is_false_positive(rec):
        docid, y, z, text = rec
        p = safe_sigmoid(z)
        return (y == 0) and (p >= best_thr)

    false_positives = test_scored.filter(is_false_positive).take(3)

    if len(false_positives) == 0:
        print("No false positives found at best F1 threshold.")
    else:
        for i, (docid, y, z, text) in enumerate(false_positives, start=1):
            snippet = text[:300].replace("\n", " ")
            print(f"False Positive #{i}:")
            print(f"  docid   = {docid}")
            print(f"  snippet = {snippet}")

    sc.stop()


if __name__ == "__main__":
    main()
