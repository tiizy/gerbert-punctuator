import pytest
from src.preprocess.dereko.split_sentence import split_raw_text

def test_split_raw_text():
    test_input = ["AP/dpa Mehr über den Konflikt unter: http://www.oau-oua.org"]
    expected_result = []
    actual_result = split_raw_text(test_input)
    print(actual_result)
    assert expected_result == actual_result