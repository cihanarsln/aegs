import pandas as pd
import numpy as np

import preprocessing
import sentence
import vectorization
import topic
import util

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def aegs(essay, selected_topic):

    # util.create_pre_calculated_result_csv(0, 1783, 'result.csv')
    # util.create_pre_calculated_result_csv(1783, 3583, 'library_result.csv')
    # util.create_pre_calculated_result_csv(3583, 5309, 'cyclist_result.csv')

    # topic.find_topic(essay, 3)

    if selected_topic == 'Computer': dataframe = pd.read_csv('result.csv', encoding="ISO-8859-1")
    elif selected_topic == 'Library': dataframe = pd.read_csv('library_result.csv', encoding="ISO-8859-1")
    elif selected_topic == 'Cyclist': dataframe = pd.read_csv('cyclist_result.csv', encoding="ISO-8859-1")

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
    X_test = sc.transform(X_test)
    # essay_test = sc.transform(essay_data)

    # Run Random Forest
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    # essay_pred = regressor.predict(essay_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

    '''
    predicted_topic = topic.find_topic(essay_without_stopwords, essay_pred[0])

    print(essay_pred)
    print(predicted_topic)
    print("finish")

    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('--------------------------------')
    for i in range(len(y_pred)):
        print(y_test[i], y_pred[i])

    print("finish")
    '''

    '''
        # This part of the code to create result.csv files these files ready for use random forest
        # Only tf-idf values must change
    '''

aegs("Dear local newspaper I raed ur argument on the computers and I think they are a positive effect on people. The first reson I think they are a good effect is because you can do so much with them like if you live in mane and ur cuzin lives in califan you and him could have a wed chat. The second thing you could do is look up news any were in the world you could be stuck on a plane and it would be vary boring when you can take but ur computer and go on ur computer at work and start doing work. When you said it takes away from exirsis well some people use the computer for that too to chart how fast they run or how meny miles they want and sometimes what they eat. The thrid reson is some peolpe jobs are on the computers or making computers for exmple when you made this artical you didnt use a type writer you used a computer and printed it out if we didnt have computers it would make ur @CAPS1 a lot harder. Thank you for reading and whe you are thinking adout it agen pleas consiter my thrie resons.", 'Computer')

