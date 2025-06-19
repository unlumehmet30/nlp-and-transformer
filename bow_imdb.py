# -*- coding: utf-8 -*-
"""
Created on Thu May 22 13:04:29 2025

@author: mhmtn
"""

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from collections import Counter

"""doc=[

     "kedi evde",
     "kedi bahçede"]
vect=CountVectorizer()
x=vect.fit_transform(doc)
feature_=vect.get_feature_names_out()
x=x.toarray()"""
df=pd.read_csv(r"C:\Users\mhmtn\Downloads\IMDB Dataset.csv (1)\IMDB Dataset.csv")
doc=df["review"]
labels=df["sentiment"]


def clean_text(text):
    text=text.lower()
    text=re.sub(r"\d+","",text)
    text=re.sub(r"[^\w\s]","",text)
    text=" ".join([word for word in text.split() if len(word)>2])
    return text
doc=[clean_text(row) for row in doc]

vect=CountVectorizer()
x=vect.fit_transform(doc[:75])
feature=vect.get_feature_names_out()
vector=x.toarray()

df_bow=pd.DataFrame(vector,columns=feature)

vord_count=x.sum(axis=0).A1
















