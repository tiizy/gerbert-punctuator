import os


def save_file(input_data, path, filename) -> str:
    """Saves input to a single file.

    Args:
        input_data (list): File with lines of text.
    Returns:
        None
    """
    with open(path + os.sep + filename, "w", encoding="utf8") as f:
        for line in input_data:
            f.write("%s\n" % line)