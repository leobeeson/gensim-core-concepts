from data.stopwords import stopwords_english

from gensim import corpora
from smart_open import open

import re


stoplist = stopwords_english["stopwords"]
pattern = re.compile("\n|\. |,|;|\?|!|\(|\)|\[|\]")

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



sample_text = "javascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\njavascript\n\njs\necmascript\n.js\njavascript-execution\nvanilla-js\nvanillajs\njavascript-library\njavascript-runtime\nvanilla-javascript\njavascript-module\nclassic-javascript\njavascript-alert\njavascript-dom\njavascript-disabled\nFor questions regarding programming in ECMAScript (JavaScript/JS) and its various dialects/implementations (excluding ActionScript). Note JavaScript is NOT the same as Java! Please include all relevant tags on your question; e.g., [node.js], [jquery], [json], [reactjs], [angular], [ember.js], [vue.js], [typescript], [svelte], etc."
new_text = re.sub("\n", " ", sample_text)