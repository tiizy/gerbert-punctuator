import re
from src.preprocess.dereko.split_sentence import split_raw_text


def process(sentences : list) -> list:
    """Removes unwanted characters and sentences from the list of sentences
    
    Args: 
        sentence (list):
    Returns: 
        regex_content (list):
    """
    regex_content = []
    for line in sentences:
        line = re.sub(r"^\s", "", line, flags=re.MULTILINE) #remove space at the beginning
        line = re.sub(r"^\w+\s\-\s", "", line, flags=re.MULTILINE) #remove city (newspaperlike)
        line = re.sub(r"^\w+:\s?|^[A-Z]{2,3}\s[A-Z]{2,4}:\s?", "", line, flags=re.MULTILINE) #remove speaker
        line = re.sub(r"^.*http.*$", "", line, flags=re.MULTILINE) #remove URL
        line = re.sub(r"^.*\.\s\.\s?\.?.*$", "", line, flags=re.MULTILINE) #remove ". . ."
        line = re.sub(r"^\d\s.*-$", "", line, flags=re.MULTILINE) #remove headlines
        line = re.sub(r"^\(.*\)\.?$", "", line, flags=re.MULTILINE) #remove lines containing only text in parentheses
        line = re.sub(r"^\).*.$|^\(.*$", "", line, flags=re.MULTILINE) #remove lines starting with opened/closed parenthesis
        line = re.sub(r"^.*-$", "", line, flags=re.MULTILINE) #remove lines ending with "-"
        line = re.sub(r"^[a-z].*$", "", line, flags=re.MULTILINE) #remove lines starting with lowercase characters
        line = re.sub(r"^\,\s[A-z].*$", "", line, flags=re.MULTILINE) #remove lines starting with ,
        line = re.sub(r"\s\(.*Tel\.?(efon)?:.*\)", "", line, flags=re.MULTILINE) #remove phone numbers
        line = re.sub(r"^\"\s?(?!.*\")", "", line, flags=re.MULTILINE) #remove unclosed quotes at the beginning of the line
        line = re.sub(r"^\"\s", "", line, flags=re.MULTILINE) #remove unclosed quotes at the beginning of the line (in case of further quotes present)
        line = re.sub(r"^\:\s|^\.$|^\[\s|^\-\s|^\"\)\s|\/\s|^\]\s", "", line, flags=re.MULTILINE) #remove further unnecessary punctuation at the beginning
        if re.search(r"^([^\r\n\"]*)\"([^\r\n\"]*)$", line, flags=re.MULTILINE) != None: #remove unclosed or unopened quotes
            line = line.replace("\"", "")
        if line != None:
            if line.endswith((".", "?", "!", ".\"")):
                regex_content.append(line)
    
    return regex_content
