import re
import numpy as np

# Load up all of the 19997 documents in the corpus.
corpus = sc.textFile("s3://luisguzmannateras/Assignment4/20_news_same_line.txt")

# Each entry in valid_lines will be a line from the text file.
valid_lines = corpus.filter(lambda x: 'id' in x)

# Now we transform it into a bunch of (docID, text) pairs.
key_and_text = valid_lines.map(
    lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:])
)

# Now we split the text in each (docID, text) pair into a list of words.
# After this, we have a data set with (docID, ["word1", "word2", "word3", ...]).
# We have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents.
regex = re.compile('[^a-zA-Z]')
key_and_list_of_words = key_and_text.map(
    lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split())
)

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
all_words = key_and_list_of_words.flatMap(lambda x: ((j, 1) for j in x[1]))

# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
all_counts = all_words.reduceByKey(lambda a, b: a + b)

# And get the top 20,000 words in a local array.
# Each entry is a ("word1", count) pair.
top_words = all_counts.top(20000, lambda x: x[1])

# And we'll create a RDD that has a bunch of (word, dictNum) pairs.
# Start by creating an RDD that has the number 0 thru 20000.
# 20000 is the number of words that will be in our dictionary.
twenty_k = sc.parallelize(range(20000))

# Now, we transform (0), (1), (2), ... to ("mostcommonword", 0) ("nextmostcommon", 1), ...
# The number will be the spot in the dictionary used to tell us where the word is located.
# HINT: make use of top_words in the lambda that you supply.
dictionary = twenty_k.map(lambda i: (top_words[i][0], i))

# Finally, print out some of the dictionary, just for debugging.
dictionary.top(10)
