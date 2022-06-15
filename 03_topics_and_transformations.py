from classes.my_dictionary import MyDictionary
from data.stopwords import stopwords_english
from smart_open import open

import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


CORPUS_FILEPATH = "data/sample_corpus_software_engineering_small.txt"
STOPWORDS = stopwords_english["stopwords"]


my_dict = MyDictionary(CORPUS_FILEPATH)
texts = [
    [
      word for word in document.lower().split() if word not in STOPWORDS
    ]
for document in open(CORPUS_FILEPATH)
]

corpus = [my_dict.dictionary.doc2bow(text) for text in texts]
