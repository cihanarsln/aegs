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
essays_scores = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
essays_scores = essays_scores.iloc[:1783, :]
essays = essays_scores['essay']

a = sentence.find_sentence_counts(essays)
b = preprocessing.remove_stopwords(essays)
c = sentence.find_word_counts(b)

a = preprocessing.remove_stopwords(essays)
b = vectorization.find_word_vector(a)
print("finish")


