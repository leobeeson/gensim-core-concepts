from data.stopwords import stopwords_english

from gensim import corpora
from smart_open import open
from typing import List

import re


PUNCTUATION_BLACKLIST = "\. |,|;|\?|!|\(|\)|\[|\]"
STOPWORDS = stopwords_english["stopwords"]


class MyDictionary:
    
    pattern = re.compile(PUNCTUATION_BLACKLIST)
    
    def __init__(self, corpus_filepath):
        self.dictionary = self.stream_dictionary(corpus_filepath)
        self.compactify_dictionary(min_freq=0)

    def stream_dictionary(self, corpus_filepath):
        dictionary = corpora.Dictionary(
            MyDictionary.tokenize(line) for line in open(corpus_filepath)
            )
        return dictionary
    
    @classmethod
    def remove_punctuation(cls, line: str) -> str:       
        clean_doc = cls.pattern.sub("", line).replace("\\n", " ")
        return clean_doc

    @classmethod
    def tokenize(cls, line: str) -> List[str]:
        tokenized_line = [
                word for word in cls.remove_punctuation(line).
                lower().
                split() if word not in STOPWORDS
                ]
        return tokenized_line

    def compactify_dictionary(self, min_freq: int) -> None:
        below_threshold_ids = [token_id for token_id, term_freq in self.dictionary.cfs.items() if term_freq <= min_freq]
        self.dictionary.filter_tokens(below_threshold_ids)
        self.dictionary.compactify()
