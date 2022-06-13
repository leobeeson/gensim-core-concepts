from stopwords import stopwords_english

from gensim.corpora import Dictionary
from collections import defaultdict
from typing import List

import json
import re


CORPUS_FILEPATH = "sample_corpus_software_engineering.json"
CORPUS_KEY = "corpus"
PUNCTUATION_BLACKLIST = "\. |,|;|\?|!|\(|\)|\[|\]"
STOPWORDS = stopwords_english["stopwords"]


def read_in_corpus(filepath: str, corpus_key: str) -> List[str]:
    with open(filepath) as json_file:
        corpus_json = json.load(json_file)
        corpus = corpus_json[corpus_key]
        return corpus


def remove_punctuation(corpus: List[str]) -> List[str]:
    pattern = re.compile(PUNCTUATION_BLACKLIST)
    clean_corpus = []
    for doc in corpus:
        clean_doc = pattern.sub("", doc)
        clean_corpus.append(clean_doc)
    return clean_corpus


def tokenize(corpus: List[str]) -> List[List[str]]:
    tokenized_corpus = [
        [word for word in document.
        lower().
        split() if word not in STOPWORDS] 
        for document in corpus
        ]
    return tokenized_corpus


def get_word_frequencies(corpus: List[List[str]]) -> defaultdict(int):
    frequencies = defaultdict(int)
    for text in corpus:
        for token in text:
            frequencies[token] += 1
    return frequencies


def filter_by_word_frequency_count(corpus: List[List[str]], frequencies: defaultdict(int)) -> List[List[str]]:
    filtered_corpus = [
        [token for token in doc if frequencies[token] > 1] for doc in corpus
    ]
    return filtered_corpus


def build_corpus_dictionary(filepath: str, key: str) -> Dictionary:
    corpus = read_in_corpus(filepath, key)
    corpus = remove_punctuation(corpus)
    corpus = tokenize(corpus)
    freqs = get_word_frequencies(corpus)
    corpus = filter_by_word_frequency_count(corpus, freqs)
    dictionary = Dictionary(corpus)    
    return dictionary


dictionary = build_corpus_dictionary(CORPUS_FILEPATH, CORPUS_KEY) 
dictionary.save("/tmp/stackexchange_corpus_dictionary.dict")


# new_doc = "python and java developer needed by UK startup"
# new_vec = dictionary.doc2bow(new_doc.lower().split())