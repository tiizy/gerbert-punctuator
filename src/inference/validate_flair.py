from flair.datasets import ClassificationCorpus
from flair.models import TextClassifier

model_path = "saved_models/flair-01-02/best-model.pt"
model = TextClassifier.load(model_path)

def flair_validation():

    data_path = "data/processed/dereko/"
    corpus = ClassificationCorpus(data_path,
                                test_file='flair_test.txt',
                                dev_file='flair_val.txt',
                                train_file='flair_train.txt',
                                label_type="class"                                       
                                )

    result = model.evaluate(corpus.test, gold_label_type=model.label_type)
    
    with open("saved_models/flair_manual_log.txt", "w") as f:
        f.write(result.detailed_results)
    
flair_validation()