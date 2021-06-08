from src.preprocess.dereko.generate_pairs_bert_classification import create_classification_pairs
from src.preprocess.utils.open_file_extension import open_files
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.save_single_file import save_file


def main():
    file_content = open_files(PROCESSED_DATA_PATH, "txt")
    pairs = create_classification_pairs(file_content)
    save_file(pairs, PROCESSED_DATA_PATH, "classification_pairs.txt")    

main()