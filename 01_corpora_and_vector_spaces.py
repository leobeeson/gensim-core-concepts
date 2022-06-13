from data.stopwords import stopwords_english

from gensim import corpora

from collections import defaultdict

import logging
import json
import re


# SOURCE:
# https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# READ IN CORPUS
with open("data/sample_corpus_software_engineering.json") as json_file:
    sample_corpus_json = json.load(json_file)
sample_corpus = sample_corpus_json["corpus"][0:10]

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

# BUILD DICTIONARY
dictionary = corpora.Dictionary(processed_corpus)
dictionary.save("/tmp/stackexchange_first_ten_tags.dict")
# print(dictionary)

# BOW VECTORS
new_doc = "python and java developer needed by UK startup"
new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec)

# BOW CORPUS
bow_corpus = [dictionary.doc2bow(token) for token in processed_corpus]
corpora.MmCorpus.serialize("/tmp/stackexchange_first_ten_tags.mm", bow_corpus)
# print(bow_corpus)

from smart_open import open  # for transparently opening remote files

class MyCorpus:
    def __iter__(self):
        for line in open("data/sample_corpus_software_engineering_small.txt"):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)

for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)