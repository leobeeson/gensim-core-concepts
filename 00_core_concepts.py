from data.stopwords import stopwords_english

from gensim import corpora
from gensim import models
from gensim import similarities

from collections import defaultdict

import re
import pprint
import json

# SOURCE:
# https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-auto-examples-core-run-core-concepts-py

# READ IN CORPUS
with open("data/sample_corpus_software_engineering.json") as json_file:
    sample_corpus_json = json.load(json_file)
sample_corpus = sample_corpus_json["sample_corpus"][0:10]

# REMOVE PUNCTUATION
pattern = re.compile("\.|,|;|\?|!|\(|\)|\[|\]")
text_corpus = []
for sample_doc in sample_corpus:
    clean_doc = pattern.sub("", sample_doc)
    text_corpus.append(clean_doc)

# TOKENIZE
stoplist = stopwords_english["stopwords"]
texts = [
    [word for word in document.lower().split() if word not in stoplist] 
    for document in text_corpus
    ]

# GET WORD FREQUENCIES
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# FILTER WORDS WITH FREQ > 1
processed_corpus = [
    [token for token in text if frequency[token] > 1] 
    for text in texts
    ]

# BUILD DICTIONARY https://radimrehurek.com/gensim/corpora/dictionary.html
dictionary = corpora.Dictionary(processed_corpus)
dictionary.token2id # dict[term: dictionary_id]
dictionary.cfs # dict[dictionary_id: term_frequency] -> term frequency within corpus
dictionary.dfs # dict[dictionary_id: doc_frequency] -> # docs in which term appears
dictionary.num_docs # int -> # of docs processed
dictionary.num_pos # int -> # of processed terms
dictionary.num_nnz # int -> ∑ of # of unique terms per doc across corpus

# BOW VECTORS
new_doc = "python and java developer needed by UK startup".lower().split()
new_vec = dictionary.doc2bow(new_doc)

# BOW CORPUS
bow_corpus = [dictionary.doc2bow(token) for token in processed_corpus]

# TF-IDF MODEL
tf_idf = models.TfidfModel(bow_corpus)
words = "java python".lower().split()
print(tf_idf[dictionary.doc2bow(words)])

# SIMILARITY INDICES
index = similarities.SparseMatrixSimilarity(tf_idf[bow_corpus], num_features=12)
query_document = 'python developer with basic knowledge of java'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tf_idf[query_bow]] #FIXME -> ERROR: Canceled future for execute_request message before replies were done. The Kernel crashed while executing code in the the current cell or a previous cell.

# Currently Unreachable due to above error: #TODO: Raise bug.
print(list(enumerate(sims)))
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)
