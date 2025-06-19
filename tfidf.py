# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:00:54 2025

@author: mhmtn
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
df=pd.read_csv(r"C:\Users\mhmtn\Downloads\IMDB Dataset.csv (1)\IMDB Dataset.csv")
doc=df["review"]
def clean_text(text):
    text=text.lower()
    text=re.sub(r"\d+","",text)
    text=re.sub(r"[^\w\s]","",text)
    text=" ".join([word for word in text.split() if len(word)>2])
    return text
doc=[clean_text(row) for row in doc]
tfidf=TfidfVectorizer()
x=tfidf.fit_transform(doc[:10])
feature=tfidf.get_feature_names_out()
vector=x.toarray()
vector=pd.DataFrame(vector,columns=feature)
tf_idf=vector.mean(axis=0)
tf_idf=tf_idf.sort_values()



