from classes.my_dictionary import MyDictionary
from data.stopwords import stopwords_english
from smart_open import open

import logging

from gensim import models
import os
import tempfile


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# SOURCE:
# https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html

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


tf_dif = models.TfidfModel(corpus)

# Calling model[corpus] only creates a wrapper around the old corpus document stream – 
# actual conversions are done on-the-fly, during document iteration. 
# We cannot convert the entire corpus at the time of calling corpus_transformed = model[corpus], 
# because that would mean storing the result in main memory, 
# and that contradicts gensim’s objective of memory-independence. 
# If you will be iterating over the transformed corpus_transformed multiple times, 
# and the transformation is costly, serialize the resulting corpus to disk first and continue using that.
corpus_tf_idf = tf_dif[corpus]
for doc in corpus_tf_idf:
  print(doc)

# Serialize a chain of transformations:
lsi_model = models.LsiModel(corpus_tf_idf, id2word=my_dict.dictionary, num_topics=100)
corpus_lis = lsi_model[corpus_tf_idf]
lsi_model.print_topics(100)

# Persist models
with tempfile.NamedTemporaryFile(prefix="model-", suffix=".lsi", delete=False) as tmp:
  lsi_model.save(tmp.name)

loaded_lsi_model = models.LsiModel.load(tmp.name)
loaded_lsi_model.print_topics(1)
os.unlink(tmp.name)

# Available Transformations:
# TF-IDF
tf_idf_model = models.TfidfModel(corpus, normalize=True)

# Latent Semantic Indexing/Analysis
lsi_model = models.LsiModel(corpus_tf_idf, id2word=my_dict.dictionary, num_topics=100)

# Random Projections
rp_model = models.RpModel(corpus_tf_idf, num_topics=500)

# LDA
lda_model = models.LdaModel(corpus, id2word=my_dict.dictionary, num_topics=100)

# Hierarchical Dirichlet Process (HDP):
hdp_model = models.HdpModel(corpus, id2word=my_dict.dictionary)
