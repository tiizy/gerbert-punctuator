import os
import re

filename = "tatoeba_german.txt"
f = open(os.path.join(os.getcwd(), "data", "raw", "tatoeba", filename), "r")
file_content = f.readlines()

content_no_id = []
for line in file_content:
    content_no_id.append(re.sub(r"^\d+\s+\w+\s+", "", line, flags = re.MULTILINE))

f = open(os.path.join(os.getcwd(), "data", "processed", "tatoeba_german.txt"), "w", encoding = "utf8")
f.writelines(content_no_id)