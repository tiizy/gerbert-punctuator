from pickle import TRUE
import pytest
import os
import shutil
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.tensor_io import save_tensors_to_file
from src.preprocess.utils.tensor_io import load_tensors
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
from transformers import BertTokenizer

def test_save_load_tensors():
    test_input1, test_input2 = "Peter, wie lange denn noch?", "Peter wie lange denn noch"
    test_input3, test_input4 = "Wann kommt unser Zug an?", "Wann kommt unser Zug an"
    combination1 = test_input1 + " " + test_input2
    combination2 = test_input3 + " " + test_input4
    result = tokenize_for_bert(test_input1, test_input2)
    result2 = tokenize_for_bert(test_input3, test_input4)
    list_of_results = [result, result2]

    save_tensors_to_file(list_of_results, os.path.join(PROCESSED_DATA_PATH, "test_save"))
    input_ids, token_type_ids, attention_mask = load_tensors(os.path.join(PROCESSED_DATA_PATH, "test_save"))
    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
    list_of_loaded = []
    for el in input_ids:
        result = tokenizer.decode(el, skip_special_tokens = True)
        list_of_loaded.append(result)
    assert combination1 == list_of_loaded[0]
    assert combination2 == list_of_loaded[1]
    shutil.rmtree(os.path.join(PROCESSED_DATA_PATH, "test_save"), ignore_errors=True)