import pytest
from src.preprocess.dereko.generate_pairs_bert_classification import create_classification_pairs


def test_create_classification_pairs():
    test_input = ["Peter, wie lange noch?"]
    expected_result = [
['<PUNCT>', 'Peter', 'wie', 'lange', 'noch'], 'None', 
['Peter', '<PUNCT>', 'wie', 'lange', 'noch'], ',',
['Peter', ',', 'wie', '<PUNCT>', 'lange', 'noch'], 'None',
['Peter', ',', 'wie', 'lange', '<PUNCT>', 'noch'], 'None',
['Peter', ',', 'wie', 'lange', 'noch', '<PUNCT>'], '?']
    actual_result = create_classification_pairs(test_input)
    assert expected_result == actual_result