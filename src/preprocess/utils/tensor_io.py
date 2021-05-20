from pickle import load
import torch
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
import os


test_input1, test_input2 = "Peter, wie lange denn noch?", "Peter wie lange denn noch"
test_input3, test_input4 = "Wann bist du so dumm geworden?", "Wann bist du so dumm geworden"

result = tokenize_for_bert(test_input1, test_input2)
result2 = tokenize_for_bert(test_input3, test_input4)
list_of_results = [result, result2]
print(list_of_results)

def save_tensors_to_file(list_of_tensors : list) -> None:
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "tensors_from_pairs"), exist_ok = True)
    
    for el in list_of_tensors:
        input_ids = list_of_tensors[el]["input_ids"]
    #token_type_ids = tensor["token_type_ids"] 
    #attention_mask = tensor["attention_mask"]
    #torch.save(input_ids, os.path.join(folder_path, "input_ids.pt"))
    #torch.save(token_type_ids, os.path.join(folder_path, "token_type_ids.pt"))
    #torch.save(attention_mask, os.path.join(folder_path, "attention_mask.pt"))

def load_tensors(folder_path : str) -> torch.tensor:    
    input_ids = torch.load(os.path.join(folder_path, "input_ids.pt"))
    token_type_ids = torch.load(os.path.join(folder_path, "token_type_ids.pt"))
    attention_mask = torch.load(os.path.join(folder_path, "attention_mask.pt"))
    return input_ids, token_type_ids, attention_mask


# for el in list_of_results:
#     save_tensors_to_file(el, os.path.join(PROCESSED_DATA_PATH, "test"))
#     print(load_tensors(os.path.join(PROCESSED_DATA_PATH, "test")))