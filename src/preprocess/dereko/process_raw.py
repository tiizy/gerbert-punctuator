import os

from src.preprocess.utils.open_file_extension import open_files
from src.preprocess.dereko.split_sentence import split_raw_text
from src.preprocess.dereko.process_sentence import process
from src.preprocess.utils.save_single_file import save_file


RAW_DATA_PATH = os.path.join(os.getcwd(), "data", "raw", "dereko")
PROCESSED_DATA_PATH = os.path.join(os.getcwd(), "data", "processed", "dereko")

def main():

    content = open_files(RAW_DATA_PATH, "quote_rundschau.txt")
    splitted_sentences = split_raw_text(content)
    processed = process(splitted_sentences)
    save_file(processed, PROCESSED_DATA_PATH, "quote_single_sentences.txt")

if __name__ == "__main__":
    main()
