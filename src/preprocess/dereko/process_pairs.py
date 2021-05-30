import os
from src.preprocess.utils.tensor_io import save_tensors_to_file
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
from tqdm import tqdm


def main():
    pairs_path = os.path.join(PROCESSED_DATA_PATH, "pairs.json")
    file_content = open_json_file(pairs_path)
    list_of_tensors = []
    pbar = tqdm(total = len(file_content))
    for idx, cont in enumerate(file_content):
       sentence1 = file_content[idx][0]
       sentence2 = file_content[idx][1]
       list_of_tensors.append(tokenize_for_bert(sentence1, sentence2))
       pbar.update(1)
    pbar.close()
    save_tensors_to_file(list_of_tensors, os.path.join(PROCESSED_DATA_PATH, "tensors_from_pairs"))
if __name__ == "__main__":
    main()