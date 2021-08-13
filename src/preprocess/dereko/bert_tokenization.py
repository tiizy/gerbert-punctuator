from transformers import BertTokenizer
import torch
from tqdm import tqdm

""" import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH """


tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased", do_lower_case = False)

def tokenize_for_bert(pairs : dict) -> torch.tensor:
    """Transforms sentence-pairs into tensors.
    Args: 
        pairs: Pairs of sentences with <PUNCT>-marker and token IDs as y.
    Returns: 
        input_ids: Tensors with tokenized sentences with special tokens and padding.
        punctuation_ids: Corresponding ID of the punctuation of a specific spot in a sentence.
        attention_masks:  Corresponding attention mask.
    """

    max_len = 0
    for el in tqdm(pairs, desc = "Determining max sentence length"):
        ## Tokenize the text and add `[CLS]` and `[SEP]` tokens
        input_ids = tokenizer.encode(el["X"], add_special_tokens=True)
        # Update the maximum sentence length
        max_len = max(max_len, len(input_ids))
    print("Max sentence length: " + str(max_len))

    input_ids = []
    attention_masks = []
    punctuation_ids = []

    for el in tqdm(pairs, desc = "Tokenizing pairs"):
        encoded = tokenizer.encode_plus(
        text = el["X"],
        add_special_tokens = True, # Add [CLS] and [SEP]
        max_length = max_len, 
        padding = "max_length", # Add [PAD]s
        return_attention_mask = True,
        return_tensors = "pt" # ask the function to return PyTorch tensors
    )
        input_ids.append(encoded["input_ids"])
        punctuation_ids.append(el["y"])
        attention_masks.append(encoded["attention_mask"])
    
    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    punctuation_ids = torch.tensor(punctuation_ids)

    return input_ids, attention_masks, punctuation_ids

""" pairs = [{'X': ['<PUNCT>', '40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', '<PUNCT>', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', '<PUNCT>', 'haben', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', '<PUNCT>', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', '<PUNCT>', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', '<PUNCT>', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', '<PUNCT>', 'sagt', 'Jaakov', 'Moses'], 'y': 1}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', ',', 'sagt', '<PUNCT>', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', ',', 'sagt', 'Jaakov', '<PUNCT>', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', ',', 'sagt', 'Jaakov', 'Moses', '<PUNCT>'], 'y': 2}, {'X': ['<PUNCT>', 'Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', '<PUNCT>', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', '<PUNCT>', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', '<PUNCT>', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', '<PUNCT>', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', '<PUNCT>', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', '<PUNCT>', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', '<PUNCT>', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '<PUNCT>', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 11}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '<PUNCT>', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 11}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', '<PUNCT>', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', 'zu', '<PUNCT>', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', 'zu', 'reichen', '<PUNCT>', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', 'zu', 'reichen', 'Leuten', '<PUNCT>'], 'y': 2}, {'X': ['<PUNCT>', 'Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', '<PUNCT>', 'er', '1960', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '<PUNCT>', '1960', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', '<PUNCT>', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', '<PUNCT>', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', '<PUNCT>', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', '<PUNCT>', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 1}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', '<PUNCT>', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', '<PUNCT>', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', '<PUNCT>', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', '<PUNCT>', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', '<PUNCT>', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', '<PUNCT>', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', '<PUNCT>', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', '<PUNCT>', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 1}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', '<PUNCT>', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', '<PUNCT>', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', '<PUNCT>', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', '<PUNCT>', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', '<PUNCT>', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', '<PUNCT>', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', '<PUNCT>', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', '<PUNCT>', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen', '<PUNCT>'], 'y': 2}]
input, attention, punct = tokenize_for_bert(pairs)
f = open(os.path.join(PROCESSED_DATA_PATH, "test_result.txt"), "w")
f.write(str(input[0]))
f.write("\n")
f.write(str(attention[0]))
f.write("\n")
f.write(str(punct[0]))
f.close()
 """