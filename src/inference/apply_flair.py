from flair.models import TextClassifier
import re
from flair.data import Sentence

model_path = "saved_models/flair-01-02/best-model.pt"
model = TextClassifier.load(model_path)



def flair_inference(sentence):

    processed_sents = []
    list_words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE) #look for whole words
    for idx, el in enumerate(list_words):
        masked_list = [el for num, el in enumerate(list_words)] #create list of words
        masked_list.insert(idx, r'[MASK]') #insert the mask between the words
        processed_sents.append(masked_list)
    list_words.append(r'[MASK]') #insert the token at the end
    processed_sents.append(list_words)
    
    final_list = []
    for item in processed_sents:
        final_list.append(' '.join(item)) #convert list to string
    
    label_list = []
    for pos, sent in enumerate(final_list):
        sentence = Sentence(sent) #create a flair sentence object
        model.predict(sentence)
        label = str(sentence.labels)[1]
        if int(label) != 0:
            print(pos)

sentence = "Dies ist ein Test"
#flair_inference(sentence)