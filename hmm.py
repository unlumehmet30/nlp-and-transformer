# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:46:55 2025

@author: mhmtn
"""

import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

"""train_data=train_data = [
    [("I", "PRON"), ("am", "VERB"), ("happy", "ADJ")],
    [("She", "PRON"), ("eats", "VERB"), ("an", "DET"), ("apple", "NOUN")],
    [("The", "DET"), ("sky", "NOUN"), ("is", "VERB"), ("blue", "ADJ")],
    [("He", "PRON"), ("reads", "VERB"), ("a", "DET"), ("book", "NOUN")],
    [("Birds", "NOUN"), ("fly", "VERB"), ("high", "ADV")],
    [("They", "PRON"), ("are", "VERB"), ("tired", "ADJ")],
    [("I", "PRON"), ("am", "VERB"), ("happy", "ADJ")],
    [("She", "PRON"), ("eats", "VERB"), ("an", "DET"), ("apple", "NOUN")],
    [("The", "DET"), ("sky", "NOUN"), ("is", "VERB"), ("blue", "ADJ")],
    [("He", "PRON"), ("reads", "VERB"), ("a", "DET"), ("book", "NOUN")],
    [("Birds", "NOUN"), ("fly", "VERB"), ("high", "ADV")],
    [("They", "PRON"), ("are", "VERB"), ("tired", "ADJ")],
    [("We", "PRON"), ("love", "VERB"), ("music", "NOUN")],
    [("You", "PRON"), ("look", "VERB"), ("great", "ADJ")],
    [("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("Dogs", "NOUN"), ("are", "VERB"), ("loyal", "ADJ")],
    [("She", "PRON"), ("writes", "VERB"), ("poems", "NOUN")],
    [("It", "PRON"), ("is", "VERB"), ("cold", "ADJ"), ("today", "NOUN")],
    [("Children", "NOUN"), ("play", "VERB"), ("outside", "ADV")],
    [("This", "DET"), ("car", "NOUN"), ("is", "VERB"), ("fast", "ADJ")],
    [("I", "PRON"), ("drink", "VERB"), ("coffee", "NOUN"), ("daily", "ADV")],
    [("She", "PRON"), ("is", "VERB"), ("the", "DET"), ("teacher", "NOUN")],
   [("He", "PRON"), ("is", "VERB"), ("the", "DET"), ("pilot", "NOUN")],
   [("They", "PRON"), ("are", "VERB"), ("the", "DET"), ("workers", "NOUN")]
]"""

nltk.download("conll2000")
train=conll2000.tagged_sents("train.txt")
test=conll2000.tagged_sents("test.txt")

trainer=hmm.HiddenMarkovModelTrainer()
tagger=trainer.train(train)
test_sentence="I like going to school".split()
tags=tagger.tag(test_sentence)
print(f"yeni cümle:{tags}")
