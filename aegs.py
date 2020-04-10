import pandas as pd
import nltk

import preprocessing
import sentence
import vectorization

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

# Read first 1783 essays and scores
essays = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
essays = essays.iloc[:1783, :]

a = preprocessing.remove_stopwords(essays)
b = vectorization.find_word_vector(a)
print("finish")


