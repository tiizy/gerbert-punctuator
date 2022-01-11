from src.preprocess.utils.json_handler import open_json_file
from random import shuffle

path = "data/processed/dereko/"
filename = "classification_pairs_filtered_tatoeba_cleaned.json"
content = open_json_file(path + filename)


result = []

for idx, el in enumerate(content):
   result.append("__label__" + str(content[idx]["y"]) + " " + ' '.join([str(word) for word in content[idx]["X"]]) + "\n")

shuffle(result)


#split 80/20
train_size = int(0.8 * len(result))
train_data = result[:train_size]
val_data = result[train_size:]


with open(path + "flair_train.txt", "w", encoding="utf8") as f:
    f.writelines(train_data)
with open(path + "flair_val.txt", "w", encoding="utf8") as f:
    f.writelines(val_data)
with open(path + "flair_test.txt", "w", encoding="utf8") as f:
    f.writelines(val_data)