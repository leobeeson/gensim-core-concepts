from numpy import sort
import requests
import io
import csv
import json

import pandas as pd

# Get into a dataframe for eye-balling:
sample_data_url = "https://raw.githubusercontent.com/leobeeson/nlp_utils/master/controlled_vocabularies/software_engineering/stackexchange_tags.csv"
sample_data_raw = requests.get(sample_data_url).content
sample_data_table = pd.read_csv(io.StringIO(sample_data_raw.decode("UTF-8")))
print(sample_data_table.head())

# Get into a map:
sample_data_csv = list(
    csv.reader(
        sample_data_raw.
        decode("UTF-8").
        splitlines(), 
        delimiter=","
    )
)
headers = sample_data_csv.pop(0)
corpus_map = {}
for row in sample_data_csv:
    tag = {}
    tag["tag_name"] = row[1]
    tag["id"] = row[0]
    tag["tag_count"] = row[2]
    tag["synonyms_count"] = row[3]
    tag["synonyms"] = row[4].split("___")
    tag["tag_description"] = row[5]
    corpus_map[row[1]] = tag

# Create Gensim-specific corpus:
corpus = []
for k, v in corpus_map.items():
    high_freq_tag = f"{k}\n"*20
    tag_synonyms = "\n".join(v["synonyms"])
    tag_description = v["tag_description"]
    doc = f"{high_freq_tag}\n{tag_synonyms}\n{tag_description}"
    corpus.append(doc) 

with open("sample_corpus_software_engineering.json", "w") as f:
    json.dump({"sample_corpus": corpus}, f, sort_keys=True, indent=4, ensure_ascii=False)
