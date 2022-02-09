from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file
import os
from collections import Counter


def count_punct_types(pairs: dict):
    list_of_punctuation = []
    for pair in pairs:
        list_of_punctuation.append(pair['y'])
    print(Counter(list_of_punctuation))

path = os.path.join(PROCESSED_DATA_PATH, "pairs_final.json")
content = open_json_file(path)
count_punct_types(content)