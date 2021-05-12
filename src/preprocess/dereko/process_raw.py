import os
import re
from src.preprocess.dereko.split_sentence import split_raw_text
from src.preprocess.dereko.process_sentence import process


path = os.getcwd() + os.sep + "data" + os.sep + "raw" + os.sep + "dereko"
filenames = os.listdir(path)
txt_files = []

for filename in filenames:
    if re.search(".txt", filename) != None:
        txt_files.append(filename)

for file in txt_files:
    f = open(path + os.sep + file, "r", encoding="utf8")
    content = f.readlines()
f.close()

def main():
    splitted_sentences = split_raw_text(content)
    processed = process(splitted_sentences)

    path = os.getcwd() + os.sep + "data" + os.sep + "processed" + os.sep + "dereko"
    with open(path + os.sep + "single_sentences.txt", "w", encoding="utf8") as f:
        for sent in processed:
            f.write("%s\n" % sent)

if __name__ == "__main__":
    main()
