from src.preprocess.dereko.generate_pairs_bert_classification import create_classification_pairs
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert
from src.preprocess.dereko.assign_tokenid_punct import assign_id
from src.punctuation_token_id import PUNCTUATION_TOKEN_ID
import os
import torch
import re
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler


def prepare_test_sentence(test_sentences):
    list_x, list_y = create_classification_pairs(test_sentences)
    return list_x, list_y

def convert_to_loader(test_sentence, list_y):
    list_y = assign_id(list_y)
    result_list = []
    result_list.append({'X': test_sentence, 'y': list_y})
    input_ids, attention_masks, punctuation_ids = tokenize_for_bert(result_list)
    dataset = TensorDataset(input_ids, attention_masks, punctuation_ids)
    loader = DataLoader(
                dataset,
                sampler = SequentialSampler(dataset), # Pull out batches sequentially.
                batch_size = 32)
    return loader, test_sentence
    

def apply_model(test_sentence, list_y):

    dataloader, list_x = convert_to_loader(test_sentence, list_y)

    model_path = os.path.join(os.getcwd(), "saved_models", "trained_model_03_09.pt")
    if torch.cuda.is_available():    
    # Tell PyTorch to use available device 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-german-cased",
        num_labels = 9, # The number of output punctuation_ids--9, multi-class task.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    model.load_state_dict(torch.load(model_path, map_location = device)) #load trained model
    model.to(device)
    model.eval() #set model to evaluation mode, print it to see the last layer
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_punctuation_ids = batch[2].to(device)
        with torch.no_grad():        
            predictions = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=None)
    for idx, tensor in enumerate(predictions["logits"]): #go through predictions
        if str(torch.argmax(tensor)) != "tensor(0)": #if you find an item that is not "None"
            for id, item in enumerate(list_x): #go through the fitting sentence
                if item == "<PUNCT>":
                    list_x[id] = PUNCTUATION_TOKEN_ID[torch.argmax(tensor).item()] #add the correct punctuation token instead of "<PUNCT>"
            return list_x
        else:
            return None


def main(test_sentences):
    list_x, list_y = prepare_test_sentence(test_sentences)
    total_sents = len(list_x) + 1
    test_sent = list_x[0]
    for i in range(total_sents):
        result = apply_model(test_sent, list_y)
        if result:
            test_sent = result
            test_sent.insert(id+2, "<PUNCT>")
        for id, word in enumerate(test_sent):
            if word == "<PUNCT>":
                test_sent.remove("<PUNCT>")
                test_sent.insert(id+1, "<PUNCT>")
                break
    test_sent.remove("<PUNCT>")
    test_sent = " ".join(test_sent)
    test_sent = re.sub(r'\s([\,\.\!\?\;\'\"\(\)\:\-](?:\s|$))', r'\1', test_sent)
    return test_sent
         

test_sentences = ["Eine echte politische Anerkennung werde es allerdings nur geben können wenn die Taliban im Einklang mit den Werten der EU handelten betonte der Spanier"]
result = main(test_sentences)
print('')
print(result)
print('')