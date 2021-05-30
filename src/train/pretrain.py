from transformers import BertGenerationEncoder
from transformers import BertGenerationDecoder
from transformers import EncoderDecoderModel
from transformers import BertTokenizer
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
from src.preprocess.utils.json_handler import open_json_file
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
import os
import tqdm


def trainBert():
    encoder = BertGenerationEncoder.from_pretrained("bert-base-german-cased")
    decoder = BertGenerationDecoder.from_pretrained("bert-base-german-cased")
    #encoder = BertGenerationEncoder.from_pretrained("bert-base-german-cased", bos_token_id=101, eos_token_id=102)
    #decoder = BertGenerationDecoder.from_pretrained("bert-base-german-cased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
    bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
    #input_ids = tokenizer('This is a long article to summarize', add_special_tokens=False, return_tensors="pt").input_ids
    #labels = tokenizer('This is a short summary', return_tensors="pt").input_ids
    pairs_path = os.path.join(PROCESSED_DATA_PATH, "pairs.json")
    file_content = open_json_file(pairs_path)
    list_of_tensors = []
    pbar = tqdm(total = len(file_content))
    for idx, cont in enumerate(file_content):
       sentence1 = file_content[idx][0]
       sentence2 = file_content[idx][1]
       list_of_tensors.append(tokenize_for_bert(sentence1, sentence2))
       pbar.update(1)
    pbar.close()

    input_ids = ""
    labels = ""

    # train...
    # add a batch of 100 sentences
    loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
    loss.backward()