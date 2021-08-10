import os
from src.preprocess.utils.json_handler import save_to_json, open_json_file
from src.preprocess.dereko.assign_tokenid_punct import assign_id
from src.preprocess.dereko.generate_pairs_bert_classification import create_classification_pairs
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH


def main():
    f = open(os.path.join(PROCESSED_DATA_PATH, "single_sentences.txt"), "r", encoding="utf8")
    file_content = f.readlines()
    f.close()
    list_x, list_y = create_classification_pairs(file_content)
    list_y = assign_id(list_y)
    result_list = []
    for i in range(len(list_y)):
        result_list.append({'X': list_x[i], 'y':list_y[i]})
    save_to_json(result_list, os.path.join(PROCESSED_DATA_PATH, "classification_pairs.json_new"))

main()