from flair.datasets import ClassificationCorpus
from flair.embeddings import FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

def start_training(path):
    
    corpus = ClassificationCorpus(path,
                                test_file='flair_test.txt',
                                dev_file='flair_val.txt',
                                train_file='flair_train.txt',
                                label_type="class"                                       
                                )

    word_embeddings = [FlairEmbeddings('de-forward'), FlairEmbeddings('de-backward')]
    document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)
    classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(label_type="class"), multi_label=True, label_type="class")
    trainer = ModelTrainer(classifier, corpus)
    trainer.train('saved_models/flair/', max_epochs=4)

path = "data/processed/dereko/"
start_training(path)