import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.open_json import open_json_file
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert


def main():
    pairs_path = os.path.join(PROCESSED_DATA_PATH, "pairs.json")
    file_content = open_json_file(pairs_path)
    for idx, cont in enumerate(file_content):
       sentence1 = file_content[idx][0]
       sentence2 = file_content[idx][1]
    encoded = tokenize_for_bert(sentence1, sentence2)
if __name__ == "__main__":
    main()