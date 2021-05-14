import json


def save_to_json(input_data, save_path) -> str:
    """Converts the input into JSON format and saves it.

    Args:
        input_data (str): Data to save.
        save_path (str): Path to the save folder.
    Returns:
        None
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False)