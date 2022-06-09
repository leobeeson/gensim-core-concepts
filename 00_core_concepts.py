from data.stopwords import stopwords_english

from collections import defaultdict

import re
import pprint
import json

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
pprint.pprint(frequency)

# FILTER WORDS WITH FREQ > 1
processed_corpus = [
    [token for token in text if frequency[token] > 1] 
    for text in texts
    ]

