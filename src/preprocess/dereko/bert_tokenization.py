from transformers import BertTokenizer
import torch
from tqdm import tqdm


def tokenize_for_bert(pairs : dict) -> torch.Tensor:
    """Transforms sentence-pairs into tensors.
    Args: 
        pairs: Pairs of sentences with <PUNCT>-marker and token IDs as y.
    Returns: 
        input_ids: Tensors with tokenized sentences with special tokens and padding.
        punctuation_ids: Corresponding ID of the punctuation of a specific spot in a sentence.
        attention_masks:  Corresponding attention mask.
    """

    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased", do_lower_case = False)

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
        attention_masks.append(encoded["attention_mask"])
        punctuation_ids.append(el["y"])
    
    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    punctuation_ids = torch.tensor(punctuation_ids)
    print("All tokens in tokenizer: " + tokenizer.all_special_tokens_extended)

    return input_ids, attention_masks, punctuation_ids