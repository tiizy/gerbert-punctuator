import os
import re
from src.preprocess.dereko.split_sentence import split_raw_text
from src.preprocess.dereko.process_sentence import process


RAW_DATA_PATH = os.path.join(os.getcwd, "data", "raw", "dereko")
PROCESSED_DATA_PATH = os.path.join(os.getcwd, "data", "processed", "dereko")

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
    
    with open(PROCESSED_DATA_PATH + os.sep + "single_sentences.txt", "w", encoding="utf8") as f:
        for sent in processed:
            f.write("%s\n" % sent)

if __name__ == "__main__":
    main()
