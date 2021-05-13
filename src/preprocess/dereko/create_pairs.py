import spacy

def create_sentence_pairs(filtered_sentences : list) -> list:
    """Copies a sentence and removes all punctuation

    Args:
        filtered_sentences (list): Cleaned full sentences
    Returns:
        paired_sentences (list): Pairs of sentences, one without punctuation
    """
    nlp = spacy.load("de_core_news_md")
    paired_sentences = []

    for line in filtered_sentences:
        doc = nlp(line)
        for sentence in doc.sents:
            temp_list = []
            for token in sentence:
                if token.is_punct == True:
                    pass
                else:
                    temp_list.append(str(token))
                no_punct_sent = " ".join(temp_list)
            sent_pair = (sentence, no_punct_sent)
            paired_sentences.append(sent_pair)

    return paired_sentences