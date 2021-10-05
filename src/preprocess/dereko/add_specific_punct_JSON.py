import collections
import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file, save_to_json
from collections import Counter
import random
from src.punctuation_token_id import PUNCTUATION_TOKEN_ID

original_filename = "temp.json"
additional_filename = "additional_training_pairs_all.json"

original_path = os.path.join(PROCESSED_DATA_PATH, original_filename)
additional_path = os.path.join(PROCESSED_DATA_PATH, additional_filename)
original_file = open_json_file(original_path)
additional_file = open_json_file(additional_path)

#PUNCTUATION_TOKEN_ID = {0:"None", 1:",", 2:".", 3:"?", 4:'"', 5:"(", 6:")", 7:":", 8:"-"}
punctuation_list = []
for pair in additional_file:
    if pair['y'] == 6:
        punctuation_list.append(pair)

for pair in punctuation_list:
    original_file.append(pair)

print(f"Added {len(punctuation_list)} pairs to the original list.")

random.shuffle(original_file)

punctuation_list = []
for pair in original_file:
    punctuation_list.append(pair['y'])

number_punct = Counter(punctuation_list)
result = {}
for i in range(9):
    result[PUNCTUATION_TOKEN_ID[i]] = int(number_punct[i])
print("Following numbers are now present in the list:")
print("")
print(result)

save_to_json(original_file, os.path.join(PROCESSED_DATA_PATH, "temp.json"))