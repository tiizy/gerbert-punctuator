from src.preprocess.utils.json_handler import open_json_file
import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
import torch


pairs = open_json_file(os.path.join(PROCESSED_DATA_PATH, "test_classification_pairs.json"))
input_ids, attention_masks, punctuation_ids = tokenize_for_bert(pairs)
save_path = os.path.join(PROCESSED_DATA_PATH, "test_tensors")
torch.save(input_ids, os.path.join(save_path, "test_input_ids.pt"))
torch.save(attention_masks, os.path.join(save_path, "test_attention_masks.pt"))
torch.save(punctuation_ids, os.path.join(save_path, "test_punctuation_ids.pt"))