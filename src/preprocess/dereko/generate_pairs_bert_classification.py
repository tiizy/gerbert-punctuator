import spacy
from tqdm import tqdm
import re


def create_classification_pairs(list_sentences):
    nlp = spacy.load("de_core_news_md")
    for sentence in tqdm(list_sentences, desc="Creating pairs"):
        list_words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        list_pairs = []
        for idx, word in enumerate(list_words):
            masked_list = [el for num, el in enumerate(list_words) if not num == idx]
            masked_list.insert(idx, "<PUNCT>")
            list_pairs.append(masked_list)
            doc = nlp(word)
            y = ""
            for el in doc:
                if el.is_punct == True:
                    y = el.text
                else:
                    y = "None"
        list_pairs.append(y)
    return list_pairs