from src.preprocess.utils.json_handler import open_json_file
import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
import torch


pairs = open_json_file(os.path.join(PROCESSED_DATA_PATH, "classification_pairs_combined_at_filtered.json"))
input_ids, attention_masks, punctuation_ids = tokenize_for_bert(pairs)
save_path = os.path.join(PROCESSED_DATA_PATH, "tensors")
torch.save(input_ids, os.path.join(save_path, "input_ids.pt"))
torch.save(attention_masks, os.path.join(save_path, "attention_masks.pt"))
torch.save(punctuation_ids, os.path.join(save_path, "punctuation_ids.pt"))