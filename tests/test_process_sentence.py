import pytest
from src.preprocess.dereko.process_sentence import process

def test_process():
    test_input = ["Kock: Nein. Wir haben allen Grund, da nicht zu verzagen.", "AP/dpa Mehr über den Konflikt unter: http://www.oau-oua.org", "Berlin - Heute wurden einige der Bewaffnete verhaftet."]
    expected_result = ['Nein. Wir haben allen Grund, da nicht zu verzagen.', 'Heute wurden einige der Bewaffnete verhaftet.']
    actual_result = process(test_input)
    assert expected_result == actual_result