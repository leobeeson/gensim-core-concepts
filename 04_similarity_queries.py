from classes.my_dictionary import MyDictionary
from data.stopwords import stopwords_english
from smart_open import open

import logging

from gensim import models
from gensim import similarities


logging.basicConfig(format="%(asctime)s : %(levelname)s %(message)s", level=logging.INFO)

# SOURCE:
# https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html

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


lsi_model = models.LsiModel(corpus, id2word=my_dict.dictionary, num_topics=100)
test_doc = "python and java developer needed by UK startup"
test_vec = my_dict.dictionary.doc2bow(test_doc.lower().split())
test_vec_lsi = lsi_model[test_vec]
print(test_vec_lsi)

index = similarities.MatrixSimilarity(lsi_model[corpus])
index.save("/tmp/software_engineer_small_corpus.index")
index = similarities.MatrixSimilarity.load("/tmp/software_engineer_small_corpus.index")

sims = index[test_vec_lsi]
print(list(enumerate(sims)))

documents = [line for line in open(CORPUS_FILEPATH)]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])

