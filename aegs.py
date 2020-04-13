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

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Read first 1783 essays and scores
essays_scores = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
essays_scores = essays_scores.iloc[:1783, :]
essays = essays_scores['essay']
scores1 = essays_scores['rater1_domain1']
scores2 = essays_scores['rater2_domain1']

# Create dataset
sentence_counts = sentence.find_sentence_counts(essays)
words_without_stopwords = preprocessing.remove_stopwords(essays)
tf_idf_values = vectorization.find_word_vector(words_without_stopwords)

# Create dataframe for Random Forest Algorithm
ds = sentence.find_word_counts(words_without_stopwords)
for i in range(len(essays)):
    ds[i][0] = sentence_counts[i]
    ds[i][19] = tf_idf_values[i]
    ds[i][20] = (scores1[i] + scores2[i])/2
df = DataFrame(ds, columns=['sentence_count', 'english_word', 'non_english_word', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBP', 'VBZ', 'other_tags', 'td_idf', 'score'])

# Random Forest Algorithm
# Split train and test sets
X = df.iloc[:, 0:19].values
y = df.iloc[:, 20].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Run Random Forest
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('--------------------------------')
for i in range(len(y_pred)):
    print(y_test[i], y_pred[i], round(y_pred[i]))

print("finish")


