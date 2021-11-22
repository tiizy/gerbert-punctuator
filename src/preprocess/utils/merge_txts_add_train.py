import os
from random import shuffle
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file, save_to_json

path = os.path.join(PROCESSED_DATA_PATH, "additional_training", "_all")

first_json = open_json_file(os.path.join(PROCESSED_DATA_PATH, "tatoeba_german_reduced.json"))
second_json = open_json_file(os.path.join(PROCESSED_DATA_PATH, "classification_pairs_filtered.json"))

def merge_txts(path:str):
    sentence_list = []
    for file in os.listdir(path):
        f = open(os.path.join(path, file), encoding="utf8")
        file_content = f.readlines()
        for line in file_content:
            line.replace("\n", "")
            sentence_list.append(line)
    shuffle(sentence_list)

    final_list = []
    for item in sentence_list:
        if item != "\n":
            final_list.append(item)

    f = open(os.path.join(path, "all_shuffled.txt"), "w", encoding="utf8")
    f.writelines(final_list)
    f.close()

def mergeJSONs(first_json:str, second_json:str):
    print(len(first_json))
    for i in second_json:
        first_json.append(i)
    print(len(first_json))
    shuffle(first_json)
    save_to_json(first_json, os.path.join(PROCESSED_DATA_PATH, "classification_pairs_filtered_tatoeba.json"))