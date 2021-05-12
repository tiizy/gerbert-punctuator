import re
import spacy


def split_raw_text(raw_text: list) -> list:
    """Splits raw text into the list of sentences.

    Args:
        raw_text (list):
    Returns:
        list_sentences (list):
    """
    raw_text = set(raw_text)
    raw_text = list(filter(None, raw_text))

    filtered_content = []
    nlp = spacy.load("de_core_news_md")

    for line in raw_text:
        doc = nlp(line)
        for sent in doc.sents:
            for token in sent:
                if sent.end > 5:
                    if sent.end and token.is_punct == True:
                        if any(re.findall(r"\"|\,|\/|\(|\:", token.text)):
                            pass
                        else:
                            filtered_content.append(str(sent))

    filtered_content = set(filtered_content)
    filtered_content = list(filter(None, filtered_content))
    return filtered_content
