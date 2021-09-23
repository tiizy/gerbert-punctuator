import os
from random import shuffle
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.save_single_file import save_file


path = os.path.join(PROCESSED_DATA_PATH, "additional_training", "_all")

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