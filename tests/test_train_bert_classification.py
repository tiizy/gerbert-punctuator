import pytest
import os
import torch
from src.train.train_bert_classification import trainBertClassification
from src.train.iterator_data_loader import load_data

def test_trainBertClassification():
    train_path = os.path.join(os.getcwd(), "tests", "training_test_data", "test_training_data.pt")
    val_path = os.path.join(os.getcwd(), "tests", "training_test_data", "test_validation_data.pt")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    train_dataloader, validation_dataloader = load_data(train_data, val_data)
    trainBertClassification(train_dataloader, validation_dataloader)

    manual_log_path = "src/train/training_logs/manual_log.json"
    assert os.path.exists(manual_log_path)