from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file
import os
from collections import Counter


def count_punct_types(pairs: dict) -> dict:
    list_of_punctuation = []
    for pair in pairs:
        list_of_punctuation.append(pair['y'])
    print(Counter(list_of_punctuation))
    #{'None': 735829, '.': 50446, ',': 45852, '-': 16909, '"': 16114, ')': 3157, '(': 3153, ':': 2973, '?': 1417, "'": 480, '/': 452, '!': 281, '&': 232, ';': 24, ']': 17, '@': 4, '%': 4, '[': 2}

path = os.path.join(PROCESSED_DATA_PATH, "classification_pairs_no_ID.json")
content = open_json_file(path)
count_punct_types(content)