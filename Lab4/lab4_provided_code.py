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
words_in_corpus = {}
 
# Generate each doc.
for doc in range(0, 50):
       
 # No words in this doc yet.
    words_in_doc = {}

    # Get the topic probabilities for this doc.
    topics_in_doc = np.random.dirichlet(beta)
    
    # Generate each of the 2000 words in this document.
    for word in range(0, 2000):
   
     # Select the topic and the word.
        which_topic = sample_value(topics_in_doc)
        which_word = sample_value(words_in_topic[which_topic])
                
     # And record the word.
        words_in_doc[which_word] = words_in_doc.get(which_word, 0) + 1
                
    # Now, remember this document.
    words_in_corpus[doc] = words_in_doc
    
# And here is the array-based LDA for use with the second two

import numpy as np
import time
 
# There are 2000 words in the corpus.
alpha = np.full(2000, .1)
 
# There are 100 topics.
beta = np.full(100, .1)
 
# This gets us the probabilty of each word happening in each of the 100 topics.
words_in_topic = np.random.dirichlet(alpha, 100)
 
# words_in_corpus[i] will give us the vector of words in document i.
words_in_corpus = np.zeros((50, 2000))
 
# Generate each doc.
for doc in range(0, 50):
    # Get the topic probabilities for this doc.
 topics_in_doc = np.random.dirichlet(beta)

    # Assign each of the 2000 words in this doc to a topic.
    words_to_topic = np.random.multinomial(2000, topics_in_doc)

    # And generate each of the 2000 words.
    for topic in range(0, 100):
        words_from_current_topic = np.random.multinomial(
            				words_to_topic[topic],
     					words_in_topic[topic]
        				)
        words_in_corpus[doc] = np.add(words_in_corpus[doc], 
                                      words_from_current_topic)
