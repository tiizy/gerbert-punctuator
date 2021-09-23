import os

from src.preprocess.dereko.split_sentence import split_raw_text
from src.preprocess.dereko.process_sentence import process
from src.preprocess.utils.save_single_file import save_file
from src.preprocess.dereko.process_raw import RAW_DATA_PATH, PROCESSED_DATA_PATH


def main(periodical : str):
    colon_path = os.path.join(RAW_DATA_PATH, periodical, "colon_" + periodical + ".txt")
    hyphen_path = os.path.join(RAW_DATA_PATH, periodical, "hyphen_" + periodical + ".txt")
    parenthesis_path = os.path.join(RAW_DATA_PATH, periodical, "parenthesis_" + periodical + ".txt")
    qm_path = os.path.join(RAW_DATA_PATH, periodical, "qm_" + periodical + ".txt")
    quote_path = os.path.join(RAW_DATA_PATH, periodical, "quote_" + periodical + ".txt")

    colon_save_path = os.path.join(PROCESSED_DATA_PATH, "additional_training", periodical)
    hyphen_save_path = os.path.join(PROCESSED_DATA_PATH, "additional_training", periodical)
    parenthesis_save_path = os.path.join(PROCESSED_DATA_PATH, "additional_training", periodical)
    qm_save_path = os.path.join(PROCESSED_DATA_PATH, "additional_training", periodical)
    quote_save_path = os.path.join(PROCESSED_DATA_PATH, "additional_training", periodical)

    with open(colon_path, "r", encoding="utf8") as f:
        colon_content = f.readlines()
    with open(hyphen_path, "r", encoding="utf8") as f:
        hyphen_content = f.readlines()
    with open(parenthesis_path, "r", encoding="utf8") as f:
        parenthesis_content = f.readlines()
    with open(qm_path, "r", encoding="utf8") as f:
        qm_content = f.readlines()
    with open(quote_path, "r", encoding="utf8") as f:
        quote_content = f.readlines()

    splitted_colon = split_raw_text(colon_content)
    splitted_hyphen = split_raw_text(hyphen_content)
    splitted_parenthesis = split_raw_text(parenthesis_content)
    splitted_qm = split_raw_text(qm_content)
    splitted_quote = split_raw_text(quote_content)

    processed_colon = process(splitted_colon)
    processed_hyphen = process(splitted_hyphen)
    processed_parenthesis = process(splitted_parenthesis)
    processed_qm = process(splitted_qm)
    processed_quote = process(splitted_quote)

    filtered_colon = []
    filtered_hyphen = []
    filtered_parenthesis = []
    filtered_qm = []
    filtered_quote = []
    for sentence in processed_colon:
        if ":" in sentence:
            filtered_colon.append(sentence)
    for sentence in processed_hyphen:
        if " - " in sentence:
            filtered_hyphen.append(sentence)
    for sentence in processed_parenthesis:
        if "(" in sentence:
            filtered_parenthesis.append(sentence)
    for sentence in processed_qm:
        if "?" in sentence:
            filtered_qm.append(sentence)
    for sentence in processed_quote:
        if '"' in sentence:
            filtered_quote.append(sentence)

    print("")
    print(periodical + " colon sentences: " + str(len(filtered_colon)))
    print(periodical + " hyphen sentences: " + str(len(filtered_hyphen)))
    print(periodical + " parenthesis sentences: " + str(len(filtered_parenthesis)))
    print(periodical + " qm sentences: " + str(len(filtered_qm)))
    print(periodical + " quote sentences: " + str(len(filtered_quote)))
    print("")        
    
    save_file(filtered_colon, colon_save_path, "colon_" + periodical + ".txt")
    save_file(filtered_hyphen, hyphen_save_path, "hyphen_" + periodical + ".txt")
    save_file(filtered_parenthesis, parenthesis_save_path, "parenthesis_" + periodical + ".txt")
    save_file(filtered_qm, qm_save_path, "qm_" + periodical + ".txt")
    save_file(filtered_quote, quote_save_path, "quote_" + periodical + ".txt")

if __name__ == "__main__":
    main("wiesbadenerkur")
