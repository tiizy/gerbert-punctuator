import os
from src.preprocess.dereko.process_raw import PROCESSED_DATA_PATH
import re
from tqdm import tqdm


load_path = os.path.join(PROCESSED_DATA_PATH, "classification_pairs.txt")
save_path = os.path.join(PROCESSED_DATA_PATH, "symbols.txt")

f = open(load_path, "r", encoding="utf8")
content = f.readlines()
f.close()

filtered_content = []
filtered_line = ""

for line in tqdm(content, desc="Extracting punctuation types"):
    filtered_line = re.sub(r"^\[.*$", "", line)
    filtered_line = re.sub(r"^None.*$", "", filtered_line)
    filtered_line = re.sub(r"\n", "", filtered_line)
    if filtered_line != "":
        filtered_content.append(filtered_line)
filtered_content = set(filtered_content)

f = open(save_path, "w", encoding="utf8")
f.writelines(filtered_content)
f.close