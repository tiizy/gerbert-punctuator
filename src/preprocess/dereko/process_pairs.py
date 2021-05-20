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
    for idx, cont in enumerate(tqdm(file_content, desc="Converting to tensors")):
       sentence1 = file_content[idx][0]
       sentence2 = file_content[idx][1]
       list_of_tensors.append(tokenize_for_bert(sentence1, sentence2))
    save_tensors_to_file(list_of_tensors)
if __name__ == "__main__":
    main()