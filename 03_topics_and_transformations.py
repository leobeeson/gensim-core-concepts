from classes.my_dictionary import MyDictionary

import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


CORPUS_FILEPATH = "data/sample_corpus_software_engineering_small.txt"


dictionary = MyDictionary(CORPUS_FILEPATH)