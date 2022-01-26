from src.preprocess.utils.json_handler import open_json_file
import os

#path = os.path.join("data", "processed", "dereko", "classification_pairs_less_punct_mask.json")
path = os.path.join("data", "processed", "tatoeba_german.txt")

#content = open_json_file(path)

with open(path, "r", encoding="utf8") as f:
    content = f.readlines()

#PUNCTUATION_TOKEN_ID = {0:"None", 1:",", 2:".", 3:"?", 4:'"', 5:"(", 6:")", 7:":", 8:"-"}

part_size = (len(content) // 4)

part1 = content[0:part_size]
part2 = content[part_size + 1:part_size * 2]
part3 = content[part_size * 2 + 1:part_size * 3]
part4 = content[part_size * 3 + 1 :part_size * 4]

print("0" + " "  + str(part_size))
print(str(part_size + 1) + " " + str(part_size * 2))
print(str(part_size * 2 + 1) + " " + str(part_size * 3))
print(str(part_size * 3 + 1) + " " + str(part_size * 4))

print(len(content))


path = os.path.join("data", "processed", "tatoeba_german_part1.txt")

with open(path, "w", encoding="utf8") as f:
    f.writelines(part1)

path = os.path.join("data", "processed", "tatoeba_german_part2.txt")

with open(path, "w", encoding="utf8") as f:
    f.writelines(part2)

path = os.path.join("data", "processed", "tatoeba_german_part3.txt")

with open(path, "w", encoding="utf8") as f:
    f.writelines(part3)

path = os.path.join("data", "processed", "tatoeba_german_part4.txt")

with open(path, "w", encoding="utf8") as f:
    f.writelines(part4)