import pytest
import torch
from src.preprocess.dereko.bert_tokenization import tokenize_for_bert


def test_tokenize_for_bert():
    test_input = [{'X': ['<PUNCT>', '40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', '<PUNCT>', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', '<PUNCT>', 'haben', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', '<PUNCT>', 'wir', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', '<PUNCT>', 'daran', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', '<PUNCT>', 'gearbeitet', 'sagt', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', '<PUNCT>', 'sagt', 'Jaakov', 'Moses'], 'y': 1}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', ',', 'sagt', '<PUNCT>', 'Jaakov', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', ',', 'sagt', 'Jaakov', '<PUNCT>', 'Moses'], 'y': 0}, {'X': ['40', 'Jahre', 'haben', 'wir', 'daran', 'gearbeitet', ',', 'sagt', 'Jaakov', 'Moses', '<PUNCT>'], 'y': 2}, {'X': ['<PUNCT>', 'Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', '<PUNCT>', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', '<PUNCT>', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', '<PUNCT>', 'Aktien', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', '<PUNCT>', 'macht', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', '<PUNCT>', 'viele', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', '<PUNCT>', 'der', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', '<PUNCT>', 'Goldman', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '<PUNCT>', 'Sachs', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 11}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '<PUNCT>', 'Partner', 'zu', 'reichen', 'Leuten'], 'y': 11}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', '<PUNCT>', 'zu', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', 'zu', '<PUNCT>', 'reichen', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', 'zu', 'reichen', '<PUNCT>', 'Leuten'], 'y': 0}, {'X': ['Die', 'Ausgabe', 'von', 'Aktien', 'macht', 'viele', 'der', 'Goldman', '-', 'Sachs', '-', 'Partner', 'zu', 'reichen', 'Leuten', '<PUNCT>'], 'y': 2}, {'X': ['<PUNCT>', 'Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', '<PUNCT>', 'er', '1960', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '<PUNCT>', '1960', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', '<PUNCT>', 'Deutscher', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', '<PUNCT>', 'Meister', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', '<PUNCT>', 'wurde', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', '<PUNCT>', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 1}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', '<PUNCT>', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', '<PUNCT>', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', '<PUNCT>', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', '<PUNCT>', 'elf', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', '<PUNCT>', 'Stammspieler', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', '<PUNCT>', 'aus', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', '<PUNCT>', 'Hamburg', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', '<PUNCT>', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 1}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', '<PUNCT>', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', '<PUNCT>', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', '<PUNCT>', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', '<PUNCT>', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', '<PUNCT>', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', '<PUNCT>', 'des', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', '<PUNCT>', 'HSV', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', '<PUNCT>', 'durchlaufen'], 'y': 0}, {'X': ['Als', 'er', '1960', 'Deutscher', 'Meister', 'wurde', ',', 'stammten', 'nicht', 'nur', 'alle', 'elf', 'Stammspieler', 'aus', 'Hamburg', ',', 'sondern', 'neun', 'hatten', 'zuvor', 'die', 'Jugendmannschaften', 'des', 'HSV', 'durchlaufen', '<PUNCT>'], 'y': 2}]
    actual_result1, actual_result2, actual_result3 = tokenize_for_bert(test_input)
    print(type(actual_result1))
    assert isinstance(type(actual_result1), type(torch.Tensor))