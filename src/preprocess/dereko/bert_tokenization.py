from transformers.models.rag.retrieval_rag import Index
from transformers import BertTokenizer


def tokenize_for_bert(sentence1, sentence2 : str) -> dict:
    """Transforms sentence-pairs into tensors.
    Args: 
        sentence1, sentence2 (str): Cleaned pairs of sentences.
    Returns: 
        encoded (dict): Numpy array.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    encoded = tokenizer.__call__(
        text = sentence1,
        text_pair = sentence2,
        add_special_tokens = True, # Add [CLS] and [SEP]
        padding = "max_length", # Add [PAD]s
        return_attention_mask = True,
        return_tensors = "np"  # ask the function to return PyTorch tensors
    )
    return encoded