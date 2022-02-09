from flair.models import TextClassifier
import re
from flair.data import Sentence
from src.punctuation_token_id import PUNCTUATION_TOKEN_ID


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
    
    temp_list = []
    for item in processed_sents:
        temp_list.append(' '.join(item)) #convert list to string
    list_words.pop() #remove the unnecesary MASK token at the end

    i = 0
    for sent in temp_list:
        sentence = Sentence(sent) #create a flair sentence object
        model.predict(sentence) #predict label
        label = str(sentence.labels)[1] #extract the numerical equivalent of the punctuation token
        if label != "]" and int(label) != 0: #if there the label is not "none"
            list_words.insert(i, PUNCTUATION_TOKEN_ID[int(label)]) #insert the label at the right position, change the number to token
            i += 1
        i += 1
            

    result = " ".join(list_words)
    result = re.sub(r'\s([\,\.\!\?\;\'\"\(\)\:\-](?:\s|$))', r'\1', result)
    return result

sentence = "Der 28 jährige war in seinem Auto eingeklemmt und musste von der Feuerwehr befreit werden"
#sentence = "Ich weiß es nicht sagte Mr Hawkins"
#sentence = "Die Taliban bestätigen dass die US Truppen keine weiteren Aktionen unternommen haben und ihre Truppen abgezogen haben"
print(flair_inference(sentence))