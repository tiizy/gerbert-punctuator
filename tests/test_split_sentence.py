import pytest
from src.preprocess.dereko.split_sentence import split_raw_text

def test_split_raw_text():
    test_input = ["Kock: Nein. Wir haben allen Grund, da nicht zu verzagen.", "Die Auseinandersetzungen waren am 6. Februar nach achtmonatiger Unterbrechung trotz einer verpflichtenden Waffenstillstandsresolution des Sicherheitsrates wieder voll entflammt. AP/dpa Mehr über den Konflikt unter: http://www.oau-oua.org"]
    expected_result = ['Wir haben allen Grund, da nicht zu verzagen.', 'Die Auseinandersetzungen waren am 6. Februar nach achtmonatiger Unterbrechung trotz einer verpflichtenden Waffenstillstandsresolution des Sicherheitsrates wieder voll entflammt.']
    actual_result = split_raw_text(test_input)
    assert expected_result == actual_result