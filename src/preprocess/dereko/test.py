from src.preprocess.utils.json_handler import open_json_file
import os

path = os.path.join("data", "processed", "dereko", "classification_pairs_less_punct_mask.json")

content = open_json_file(path)

#PUNCTUATION_TOKEN_ID = {0:"None", 1:",", 2:".", 3:"?", 4:'"', 5:"(", 6:")", 7:":", 8:"-"}

print(len(content))
for item in content[877300:877346]:
    print(item)