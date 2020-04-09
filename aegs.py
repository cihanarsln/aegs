import numpy as np
import pandas as pd
import nltk
import re
import enchant

import preprocessing
import sentence

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(max_features=2000)

# Read first 1783 essays and scores
essays = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
essays = essays.iloc[:1, :]
#essays = essays.iloc[:1783, :]


def find_countVector():
    x = cv.fit_transform(essays['essay']).toarray()
    return x

a = preprocessing.remove_stopwords(essays)
b = sentence.find_word_count(a)
print("dss")


