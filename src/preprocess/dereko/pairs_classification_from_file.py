import os
from src.preprocess.utils.json_handler import save_to_json, open_json_file
from src.preprocess.dereko.assign_tokenid_punct import assign_id
from src.preprocess.dereko.generate_pairs_bert_classification import create_classification_pairs
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH

def main():
    file1 = open_json_file(os.path.join(PROCESSED_DATA_PATH, "classification_pairs_less_punct.json"))
    print(len(file1))

    file2 = open_json_file(os.path.join(PROCESSED_DATA_PATH, "additional_training_pairs_all.json"))
    print(len(file2))
    print(file2[165])
    print(file2[216])
    print(file2[166])
    print(file2[217])

    half = int(len(file2) / 2)

    for pair in file2[:half]:
        file1.append(pair)
    print(len(file1))

    save_to_json(file1, os.path.join(PROCESSED_DATA_PATH, "classification_pairs_combined_at.json"))

""" def main():
    f = open(os.path.join(PROCESSED_DATA_PATH, "additional_training", "all_shuffled_6.txt"), "r", encoding="utf8")
    file_content = f.readlines()
    f.close()
    list_x, list_y = create_classification_pairs(file_content)
    list_y = assign_id(list_y)
    result_list = []
    for i in range(len(list_y)):
        result_list.append({'X': list_x[i], 'y':list_y[i]})
    save_to_json(result_list, os.path.join(PROCESSED_DATA_PATH, "additional_training_pairs_6.json")) """

main()