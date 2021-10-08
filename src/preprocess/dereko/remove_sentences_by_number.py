import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file, save_to_json
from collections import Counter
from src.punctuation_token_id import PUNCTUATION_TOKEN_ID

filename = "temp_filtered.json"
path = os.path.join(PROCESSED_DATA_PATH, filename)
file = open_json_file(path)

#PUNCTUATION_TOKEN_ID = {0:"None", 1:",", 2:".", 3:"?", 4:'"', 5:"(", 6:")", 7:":", 8:"-"}
number_to_remove = 3
token_id = 0
count = 1

reduced_list = file
for idx, pair in enumerate(file):
    temp = pair['y']
    if count <= number_to_remove:
        if temp == token_id:
            del reduced_list[idx]
            count += 1

save_to_json(reduced_list, os.path.join(PROCESSED_DATA_PATH, "classification_pairs_filtered.json"))