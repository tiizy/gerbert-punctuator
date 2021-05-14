import os
import re


def open_files(path, extension) -> str:
    """Converts the input into JSON format and saves it.

    Args:
        path (str): Path to the folder.
        extension (str): File-extension to look for inside the folder. 
    Returns:
        content (str): The content of the files.
    """
    filenames = os.listdir(path)
    files = []

    for filename in filenames:
        if re.search(extension, filename) != None:
            files.append(filename)

    for file in files:
        f = open(os.path.join(path, file), "r", encoding="utf8")
        content = f.readlines()
    f.close()
    return content