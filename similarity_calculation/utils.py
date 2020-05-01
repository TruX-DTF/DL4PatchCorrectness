# -*- coding: utf-8 -*-

import re
from nltk.tokenize import word_tokenize

def my_split(text):
    tokens = []
    text = remove_special_chars(text)
    tmp = word_tokenize(text)
    for t in tmp:
        t = camel_case_split(t)
        tokens.extend(t)
    return tokens


def strip_comments(text):
    return re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    
def remove_special_chars(text):
    return re.sub(r"[^a-zA-Z0-9]+", ' ', text)

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0).lower() for m in matches]

if __name__ == '__main__':
    pass