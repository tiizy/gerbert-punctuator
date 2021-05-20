from src.preprocess.utils.open_json import open_json_file
import os
from transformers.models.rag.retrieval_rag import Index
from transformers import BertTokenizer

PAIRS_JSON_PATH = os.path.join(os.getcwd(), "data", "processed", "dereko", "pairs.json")

def tokenize_for_bert(sentence1, sentence2 : str) -> dict:
    """Transforms sentence-pairs into tensors.
    Args: 
        sentence1, sentence2 (str): Cleaned pairs of sentences.
    Returns: 
        encoded (dict): Pytorch tensors.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    encoded = tokenizer.__call__(
        text = sentence1,
        text_pair = sentence2,
        add_special_tokens = True, # Add [CLS] and [SEP]
        padding = "max_length", # Add [PAD]s
        return_attention_mask = True,
        return_tensors = "pt"  # ask the function to return PyTorch tensors
    )
    return encoded