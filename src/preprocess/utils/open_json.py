import json

def open_json_file(path) -> str:
    """Properly opens a JSON-file and returns its content.

    Args:
        path (str): Path to the file.
    Returns:
        content (str): The content of the file.
    """
    with open(path, "r") as f:
        file_content = json.load(f)
    return file_content