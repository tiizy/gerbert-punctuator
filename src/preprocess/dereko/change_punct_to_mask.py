from src.preprocess.utils.json_handler import open_json_file, save_to_json
import os

path = os.path.join("data", "processed", "dereko", "classification_pairs_less_punct.json")
content = open_json_file(path)

temp_list = []

for pair in content: 
    temp_dict = {}
    x_list = []
    for el in pair['X']:
        if el in "<PUNCT>":
            el = "[MASK]"
        x_list.append(el)
        temp_dict['X'] = x_list
        temp_dict['y'] = pair['y']
    temp_list.append(temp_dict)

saving_path = os.path.join("data", "processed", "dereko", "classification_pairs_less_punct_mask.json")
save_to_json(temp_list, saving_path)