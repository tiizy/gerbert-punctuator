import collections
import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file
from collections import Counter
from src.punctuation_token_id import PUNCTUATION_TOKEN_ID

filename = "classification_pairs_filtered.json"
path = os.path.join(PROCESSED_DATA_PATH, filename)
file = open_json_file(path)

punctuation_list = []
for pair in file:
    punctuation_list.append(pair['y'])

number_punct = Counter(punctuation_list)

result = {}
for i in range(9):
    #result[PUNCTUATION_TOKEN_ID[i]] = int(number_punct[i] * 0.8)
    result[PUNCTUATION_TOKEN_ID[i]] = int(number_punct[i])
print(result)