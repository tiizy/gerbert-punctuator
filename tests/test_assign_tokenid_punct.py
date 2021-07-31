import pytest
from src.preprocess.dereko.assign_tokenid_punct import assign_id

def test_assign_id():
    test_input = ['None', '!', '?', ';', "'", '"', '(', ')', ':', '-', ']', '%', '@', '&', "#", '.']
    expected_result = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 2]
    actual_result = assign_id(test_input)
    assert actual_result == expected_result