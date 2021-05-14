import os


def save_file(input_data, path, filename) -> str:
    """Saves input to a single file.

    Args:
        input_data (list): File with lines of text.
        path (str): Path to the save folder.
        filename (str): Name of the file.
    Returns:
        None
    """
    with open(os.path.join(path, filename), "w", encoding="utf8") as f:
        for line in input_data:
            f.write("%s\n" % line)