import matplotlib
from src.preprocess.utils.json_handler import open_json_file, save_to_json
import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
import matplotlib.pyplot as plt

pairs = open_json_file(os.path.join(PROCESSED_DATA_PATH, "classification_pairs_combined_at.json"))
filtered_list = []

allowed_sentence_length = 101

#length_list = []
#for idx, pair in enumerate(pairs):
#    length_list.append(len(pairs[idx]["X"]))


#plt.hist(length_list, histtype="bar", bins=100)
#plt.savefig('plot.png')
#plt.close()

counter = 0
for idx, pair in enumerate(pairs):
    if len(pairs[idx]["X"]) < allowed_sentence_length:
        filtered_list.append(pair)
        counter += 1

print(counter)
plt.hist(filtered_list, histtype="bar", bins=100)
plt.savefig('plot.png')
plt.close()


#save_to_json(filtered_list, os.path.join(PROCESSED_DATA_PATH, "classification_pairs_combined_at_filtered.json"))