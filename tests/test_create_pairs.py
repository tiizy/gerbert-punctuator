import pytest
from src.preprocess.dereko.create_pairs import create_sentence_pairs

def test_create_sentence_pairs():
    test_input = ['"Ich mag Cowboyfilme", sagt er, und Western sind doch auch nichts anderes als Heimatfilme.', 'Zum Preis wurden keine Angaben gemacht.']
    expected_result = [
        ['"Ich mag Cowboyfilme", sagt er, und Western sind doch auch nichts anderes als Heimatfilme.',
        'Ich mag Cowboyfilme sagt er und Western sind doch auch nichts anderes als Heimatfilme'],
        ['Zum Preis wurden keine Angaben gemacht.',
        'Zum Preis wurden keine Angaben gemacht']
    ]
    actual_result = create_sentence_pairs(test_input)
    assert expected_result == actual_result