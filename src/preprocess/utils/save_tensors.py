
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
import os
import json


test_input1, test_input2 = "Peter, wie lange denn noch?", "Peter wie lange denn noch"
test_input3, test_input4 = "Wann bist du so dumm geworden?", "Wann bist du so dumm geworden"

result = tokenize_for_bert(test_input1, test_input2)
result2 = tokenize_for_bert(test_input3, test_input4)
list_of_results = [result, result2]

def save_tensors_to_file(tensor : str) -> None:
    with open(os.path.join(PROCESSED_DATA_PATH, "test.json"), 'a', encoding='utf-8') as f:
        json.dump(tensor, f, ensure_ascii=False)

for el in list_of_results:
    save_tensors_to_file(str(el))