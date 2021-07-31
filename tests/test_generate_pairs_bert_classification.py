import pytest
from src.preprocess.dereko.generate_pairs_bert_classification import create_classification_pairs, remove_punct_after_marker


def test_create_classification_pairs():
    test_input = ["Peter, wie lange noch?"]
    expected_result = [
['<PUNCT>', 'Peter', 'wie', 'lange', 'noch'], 
['Peter', '<PUNCT>', 'wie', 'lange', 'noch'],
['Peter', ',', 'wie', '<PUNCT>', 'lange', 'noch'],
['Peter', ',', 'wie', 'lange', '<PUNCT>', 'noch'],
['Peter', ',', 'wie', 'lange', 'noch', '<PUNCT>']]
    actual_result_x, actual_result_y = create_classification_pairs(test_input)
    assert expected_result == actual_result_x

def test_remove_punct_after_marker():
    test_input = ['Was', 'er', 'offenbar', 'nicht', 'mitgekriegt', 'hat', ',', '<PUNCT>', 'war', ',', 'daß', 'an', 'diesem', 'Abend', 'alle', 'verloren', 'hatten', '.']
    expected_result = ['Was', 'er', 'offenbar', 'nicht', 'mitgekriegt', 'hat', ',', '<PUNCT>', 'war', 'daß', 'an', 'diesem', 'Abend', 'alle', 'verloren', 'hatten']
    actual_result = remove_punct_after_marker(test_input)
    assert expected_result == actual_result