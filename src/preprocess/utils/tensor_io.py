import os
from tqdm import tqdm
import numpy as np


def save_tensors_to_file(list_of_tensors : list, path : str) -> None:
    os.makedirs(path, exist_ok = True)
    input_ids = []
    token_type_ids = []
    attention_mask = []
    for indx, el in enumerate(tqdm(list_of_tensors, desc="Saving tensors")):
        input_ids.append(list_of_tensors[indx].get("input_ids"))
        token_type_ids.append(list_of_tensors[indx].get("token_type_ids"))
        attention_mask.append(list_of_tensors[indx].get("attention_mask"))
    np.save(os.path.join(path, "input_ids.npy"), input_ids)
    np.save(os.path.join(path, "token_type_ids.npy"), token_type_ids)
    np.save(os.path.join(path, "attention_mask.npy"), attention_mask)


def load_tensors(folder_path : str) -> np.array:    
    raw_ids = np.load(os.path.join(folder_path, "input_ids.npy"))
    input_ids = []
    for el in raw_ids:
        for i in el:
            input_ids.append(i)
    raw_token_type_ids = np.load(os.path.join(folder_path, "token_type_ids.npy"))
    token_type_ids = []
    for el in raw_token_type_ids:
        for i in el:
            token_type_ids.append(i)
    raw_attention_mask = np.load(os.path.join(folder_path, "attention_mask.npy"))
    attention_mask = []
    for el in raw_attention_mask:
        for i in el:
            attention_mask.append(i)
    return input_ids, token_type_ids, attention_mask
