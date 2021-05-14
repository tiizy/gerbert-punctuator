import os
import re

from src.preprocess.dereko.split_sentence import split_raw_text
from src.preprocess.dereko.process_sentence import process
from src.preprocess.dereko.create_pairs import create_sentence_pairs
from src.preprocess.utils.json_handler import save_to_json
from src.preprocess.utils.save_single_file import save_file


RAW_DATA_PATH = os.path.join(os.getcwd(), "data", "raw", "dereko")
PROCESSED_DATA_PATH = os.path.join(os.getcwd(), "data", "processed", "dereko")

def main():
    filenames = os.listdir(RAW_DATA_PATH)
    txt_files = []

    for filename in filenames:
        if re.search(".txt", filename) != None:
            txt_files.append(filename)

    for file in txt_files:
        f = open(RAW_DATA_PATH + os.sep + file, "r", encoding="utf8")
        content = f.readlines()
    f.close()

    splitted_sentences = split_raw_text(content)
    processed = process(splitted_sentences)
    pairs = create_sentence_pairs(processed)
    save_file(processed, PROCESSED_DATA_PATH, "single_sentences.txt")
    save_to_json(pairs, os.path.join(PROCESSED_DATA_PATH, "pairs.json"))

if __name__ == "__main__":
    main()
