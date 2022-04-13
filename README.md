# (Ger)BERT Punctuator
An automatic punctuation restoration for German, created as a part of my master thesis. The model allows the restoration of following punctuation types:

```
, . ? “ ( ) : -
```

## Libraries
The project is created with following libraries, which also can be found in the requirements.txt for a docker.

* Scrapy version: 2.5.0
* Pytest version: 6.2.3
* Spacy version: 3.0.6
* Tqdm version: 4.60.0
* Transformers version: 4.6.0
* Matplotlib version: 3.4.2
* Tensorboard version: 2.6.0
* Torchmetrics version: 0.5.0 
* Flair version: 0.10

## Setup
I strongly recommend to use docker in order to download all existing libraries. This section contains step-by-step instructions for training the BERT-model on your data.

## Preprocessing the data

### 1. Data
Your training data should be in the .txt-format, containing full German sentenceы with correct punctuation.

### 2. process_raw.py
Converts the data into a file with just one sentence per line, removing very short (< 5 words) sentences. A set of Regular Expressions cleans up the sentences. Outputs a .txt-file.

### 3. pairs_classification_from_file.py
Using the previously created .txt, it converts the cleaned up sentences for a BERT classification task. The output is a JSON.

### 4. remove_long_sents.py
You have to make sure, your sentences contain < 125 words (for BERT), they can be filtered using this file.

### 5. check_punctuation_numbers.py (optional)
With this file you can check the number of sentences in each punctuation category.

### 6. add_specific_punct_JSON.py (optional)
If you need to add more data (I trained the model using about a million sentences), you can use this file to do so.

### 7. remove_sentence_by_number.py (optional)
In case the data is not balanced, you can specify a number of sentences of each punctuation category that should be deleted using this file.

### 8. save_tokenized_classification_pairs.py
Adds all necessary BERT tokens into your data and converts it into numbers. It also creates an attention mask for each sentence. Outputs 3 PyTorch files, including the sentences, the corresponding attention masks and punctuation.

### 9. split_training_validation.py
Splits the data into training (80% of data) and validation (20%) sets. Outputs the sets in TensorDataset-format.

## Training
A CUDA-capable GPU is needed in order to train the model in reasonable period of time. Using a 2070 Ti each epoch took about 90 minutes.

### train_bert_classification.py
Trains the model with settings recommended by BERT authors for 4 epochs. Tensorboard and Torchmetrics is used to log the results, additionally, a manual log is created.

## Inference
After a successful training, the created model can be assessed using the described file.

### apply_trained_model.py
You can enter a single sentences (at the bottom of the file) in order to test the model.
