import json


def save_to_json(input_data, save_path) -> str:
    """Converts the input into JSON format and saves it.

    Args:
        input_data (str): Data to save.
        save_path (str): Path to the save folder.
    Returns:
        None
    """
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(input_data, f, ensure_ascii=False)

def open_json_file(path) -> str:
    """Properly opens a JSON-file and returns its content.

    Args:
        path (str): Path to the file.
    Returns:
        content (str): The content of the file.
    """
    with open(path, "r", encoding="utf8") as f:
        file_content = json.load(f)
    return file_content