import pandas as pd
import numpy as np

import preprocessing
import sentence
import vectorization
import topic

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def predict(essay, selected_topic):
    if selected_topic == 'Computer':
        dataframe = pd.read_csv('result.csv', encoding="ISO-8859-1")
    elif selected_topic == 'Library':
        dataframe = pd.read_csv('library_result.csv', encoding="ISO-8859-1")
    elif selected_topic == 'Cyclist':
        dataframe = pd.read_csv('cyclist_result.csv', encoding="ISO-8859-1")

    essays_without_stopwords = dataframe.iloc[:, 26].values

    essay = [essay]
    essay_sentence_counts = sentence.find_counts(essay)
    essay_words_without_stopwords = preprocessing.remove_stopwords(essay)
    essay_words_without_stopwords_count = len(essay_words_without_stopwords[0])
    essay_without_stopwords = ' '.join(essay_words_without_stopwords[0])
    essays_without_stopwords = np.append(essays_without_stopwords, essay_without_stopwords)
    tf_idf_scores = vectorization.find_word_vector_v2(essays_without_stopwords)

    for i in range(len(dataframe)):
        dataframe.iloc[i, 23] = tf_idf_scores[i][0]

    essay_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(essay_sentence_counts[0])):
        essay_data[i] = essay_sentence_counts[0][i]
    essay_data[22] = essay_words_without_stopwords_count
    essay_data[23] = tf_idf_scores[-1][0]
    essay_data[24] = tf_idf_scores[-1][1]
    essay_data = [np.array(essay_data)]

    """ Random Forest Algorithm """
    # Split train and test sets
    X = dataframe.iloc[:, 0:25].values
    y = dataframe.iloc[:, 25].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    essay_test = sc.transform(essay_data)

    # Run Random Forest
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    essay_score_pred = regressor.predict(essay_test)[0]
    ''' End of the Random Forest '''

    # Predict Topic
    essay_topic_pred = topic.find_topic(essay_words_without_stopwords)

    if selected_topic != str(essay_topic_pred).capitalize():
        print('Selected topic and predicted topic did not match. Are you sure you selected right topic?')
        print('Predicted Topic: ', str(essay_topic_pred).capitalize())
        essay_score_pred *= 0.6
    print('Predicted Score: ', essay_score_pred)

def visualize():

    dataframe = pd.read_csv('result.csv', encoding="ISO-8859-1")


    """ Random Forest Algorithm """
    # Split train and test sets
    X = dataframe.iloc[:, 0:25].values
    y = dataframe.iloc[:, 25].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Run Random Forest
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print('---------- Score Scores ----------')
    print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))