import pytest
import transformers
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert


def test_tokenize_for_bert():
    test_input1, test_input2 = "Peter, wie lange denn noch?", "Peter wie lange denn noch"
    actual_result = tokenize_for_bert(test_input1, test_input2)
    assert isinstance(type(actual_result), type(transformers.tokenization_utils_base.BatchEncoding))