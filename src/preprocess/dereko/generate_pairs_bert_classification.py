import spacy
from tqdm import tqdm
import re


def create_classification_pairs(list_sentences : list) -> list:
    """Prepares list of single sentences for classification task

    Args:
        list_sentences (list): Cleaned full sentences
    Returns:
        list_pairs (list): Pairs including x and y, where x is a sentence and y is a punctuation on a specific spot.
    """
    nlp = spacy.load("de_core_news_md")
    list_pairs = []
    for sentence in tqdm(list_sentences, desc="Creating pairs"):
        list_words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE) #look for whole words
        i = 0
        while i < len(list_words): 
            doc = nlp(list_words[i]) #create spacy object
            masked_list = [el for num, el in enumerate(list_words)] #create list of words
            masked_list.insert(i, "<PUNCT>") #insert the mask between the words
            list_pairs.append(masked_list)
            y = ""
            for el in doc:
                if el.is_punct == True:
                    y = el.text
                    i += 1 #going to the next space, to prevent searching after a punctuation has been found
                else:
                    y = "None"
            list_pairs.append(y)
            i += 1
    for idx, pair in enumerate(tqdm(list_pairs, desc="Removing punctuation after marker")):
        if idx % 2 == 0: #check only the pair with the sentence
            for idx, word in enumerate(pair):
                if word == "<PUNCT>":
                    marker_idx = idx
                if idx > marker_idx:
                    doc = nlp(word)
                    for el in doc:
                        if el.is_punct == True:
                            pair.remove(word)
    return list_pairs