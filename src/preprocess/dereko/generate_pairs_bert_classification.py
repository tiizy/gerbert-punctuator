import spacy
from tqdm import tqdm
import re
from src.punctuation_token_id import PUNCTUATION_TOKEN_ID


nlp = spacy.load("de_core_news_md")

def create_classification_pairs(list_sentences : list) -> list:
    """Prepared list of single sentences for classification task

    Args:
        list_sentences (list): Cleaned full sentence
    Returns:
        list_x (list): Pair x, which is a sentence
        list_y (list): Pair y, which is a punctuation on a specific spot of pair x.
    """

    list_pairs = []
    for sentence in tqdm(list_sentences, desc="Creating pairs"):
        list_words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE) #look for whole words
        i = 0
        while i < len(list_words): 
            masked_list = [el for num, el in enumerate(list_words)] #create list of words
            masked_list.insert(i, r'[MASK]') #insert the mask between the words
            masked_list = remove_punct_after_marker(masked_list)
            list_pairs.append(masked_list)           
            y = ""
            for el in list_words[i]:
                if el in PUNCTUATION_TOKEN_ID.values():
                    y = el
                    i += 1 #going to the next space, to prevent searching after a punctuation has been found
                else:
                    y = "None"
            list_pairs.append(y)
            i += 1
    list_x = list_pairs[::2]
    list_y = list_pairs[1::2]
    return list_x, list_y


def remove_punct_after_marker(sentence :list) -> list:
    """Removes any punctuation after the [MASK]-token.
    Args:
        sentence (list): Sentence with [MASK]-token and punctuation.
    Returns:
        processed_sentence (list): Sentence in which any punctuation after the [MASK]-token is removed.
    """
    processed_sentence = []
    is_punct_found = False
    for word in sentence:
        if word == r'[MASK]':
            is_punct_found = True
        
        if is_punct_found:
            doc = nlp(word)
            if str(doc) == r"[MASK]" or doc[0].is_punct == False:
                processed_sentence.append(word)
        else:
            processed_sentence.append(word)
    return processed_sentence