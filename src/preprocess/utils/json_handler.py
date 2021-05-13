import json

def save_to_json(input_data, save_path) -> str:
    """Converts the input into JSON format.

    Args:
        input_data:
    Returns:
        None
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False)