from data.stopwords import stopwords_english

from gensim import corpora
from smart_open import open

import re


stoplist = stopwords_english["stopwords"]
pattern = re.compile("\. |,|;|\?|!|\(|\)|\[|\]")

dictionary = corpora.Dictionary()

class MyCorpus:
    def __iter__(self):
        for line in open("data/sample_corpus_software_engineering_small.txt"):
            clean_line = pattern.sub(" ", line)
            preprocessed_line = [word for word in clean_line.lower().split() if word not in stoplist] 
            yield dictionary.doc2bow(preprocessed_line)

corpus = MyCorpus()
print(corpus)

for vector in corpus:
    print(vector)



