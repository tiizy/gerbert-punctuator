
from src.punctuation_token_id import PUNCTUATION_TOKEN_ID

def assign_id(list_y : list) -> list:
    """Changes the punctuation strings to the matching tokenID.

    Args:
        list_y (list): Pair y, which is a string with the punctuation on a specific spot of pair x.
    Returns:
        list_id (list): Pair y with tokenIDs instead of strings.
    """
    list_id = []
    important_chars = ["None", ",", ".", "?", '"', "(", ")", ":", "-"]

    for punct in list_y:
        if punct not in important_chars:
            list_id.append(0)
            continue
        for id, string in PUNCTUATION_TOKEN_ID.items():
            if punct == string:
                list_id.append(id)

    return list_id