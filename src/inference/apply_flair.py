from flair.models import TextClassifier
from flair.datasets import ClassificationCorpus
import re
from flair.data import Sentence
from flair.datasets import DataLoader

model_path = "saved_models/flair-01-02/best-model.pt"
model = TextClassifier.load(model_path)

data_path = "data/processed/flair/"
corpus = ClassificationCorpus(data_path,
                            test_file='flair_test.txt',
                            dev_file='flair_val.txt',
                            train_file='flair_train.txt',
                            label_type="class"                                       
                            )
#test_dataloader = DataLoader(corpus.test, batch_size=32)

result, main_score = model.evaluate(corpus.test, gold_label_type=model.label_type)
print(result.detailed_results)

def flair_inference(sent_list):
    processed_sents = []
    for sentence in sent_list:
        list_words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE) #look for whole words
        i = 0
        while i < len(list_words):
            masked_list = [el for num, el in enumerate(list_words)] #create list of words
            masked_list.insert(i, r'[MASK]') #insert the mask between the words   
        processed_sents.append(masked_list)
    final_list = []
    for item in processed_sents:
        final_list.append(' '.join(item))
    print(final_list)


#sentence = Sentence("Wann wirst du da sein [MASK]")

#model.predict(sentence)
#print(sentence.labels)

sent_list = ["Wann wirst du da sein [MASK]", "Dies ist ein Test"]
