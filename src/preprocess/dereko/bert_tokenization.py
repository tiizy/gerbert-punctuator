from src.preprocess.utils import open_file_extension
import os
from transformers import BertTokenizer


#json
f = open(os.path.join("data", "processed", "dereko", "pairs.json"))
file_content = f.readlines()
f.close()

print(file_content[0][2])


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sentence = "Peter, wie lange denn noch?"
sentence2 = "Peter wie lange denn noch"

encoded = tokenizer.encode_plus(
    text = sentence,  # the sentence to be encoded
    text_pair = sentence2,
    add_special_tokens = True,  # Add [CLS] and [SEP]
    padding = "max_length",  # Add [PAD]s
    return_attention_mask = True,  # Generate the attention mask
    truncation = True,
    return_tensors = 'np'  # ask the function to return PyTorch tensors
)
#google how to save either np or pt
print(encoded["input_ids"])