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

def aegs(essay):

    # create_pre_calculated_result_csv(0, 1783, 'result.csv')
    # util.create_pre_calculated_result_csv(1783, 3583, 'library_result.csv')
    # util.create_pre_calculated_result_csv(3583, 5309, 'cyclist_result.csv')

    dataframe = pd.read_csv('result.csv', encoding="ISO-8859-1")
    essays_without_stopwords = dataframe.iloc[:, 25].values

    essay = [essay]
    essay_sentence_counts = sentence.find_counts(essay)
    essay_words_without_stopwords = preprocessing.remove_stopwords(essay)
    essay_words_without_stopwords_count = len(essay_words_without_stopwords[0])
    essay_without_stopwords = ' '.join(essay_words_without_stopwords[0])
    essays_without_stopwords = np.append(essays_without_stopwords, essay_without_stopwords)
    tf_idf_scores = vectorization.find_word_vector_v2(essays_without_stopwords)

    for i in range(len(dataframe)):
        dataframe.iloc[i, 23] = tf_idf_scores[i]

    essay_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(essay_sentence_counts[0])):
        essay_data[i] = essay_sentence_counts[0][i]
    essay_data[22] = essay_words_without_stopwords_count
    essay_data[23] = tf_idf_scores[len(tf_idf_scores)-1]
    essay_data = [np.array(essay_data)]


    """ Random Forest Algorithm """
    # Split train and test sets
    X = dataframe.iloc[:, 0:24].values
    y = dataframe.iloc[:, 24].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    essay_test = sc.transform(essay_data)

    # Run Random Forest
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    # y_pred = regressor.predict(X_test)
    essay_pred = regressor.predict(essay_test)

    essay_topic = topic.find_topic(essay_without_stopwords, essay_pred[0])

    print(essay_pred)
    print(essay_topic)
    print("finish")

    '''
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

aegs("Certain materials being removed from libraries such as books, music and magazines, shouldn't be removed from the libraries. It gives people a chance to understand how the real world @CAPS2.     Having certain materials such as books and music definitly should not be removed, because most books and music can show most people how bad the statement in the book @CAPS2 or how bad the lyrics are in a song, and help that person to avoid that type of thing that the book or song @CAPS2 saying to the reader or listener. People should give every type of music at least a try and not always doubt what they hear about what people say about that type of music. I always hear about people saying how bad the band @PERSON1 A.M. @CAPS2, just because in the lyrics it talks about drugs and how much cursing each song has. Really the band @CAPS2 talking about one mans life and how he turns his life from being a drug addict to having the best life someone could ever live. People always doubted him and never gave his music a chance. Another example would be @PERSON1's book, '@CAPS1 @CAPS2 @CAPS3 @CAPS4' for it talks about drug addicts, homeless people, people who have been born with disfigured arms or even someone who lost there legs, and telling how beautiful each and everyone of them really are. His book taught me a few things and made me think different about people. It doesn't matter how they look or how they talk, no matter what, that person @CAPS2 beautiful.     As far as movies and magazines has gone within the last few years, I think that the also shouldn't be taken from libraries. I think @CAPS1 for the same reason of how I feel about the books and music. Of course we see previews of movies and think that they @MONTH1 not be good, but libraries shouldn't keep leave them out. Movies @CAPS2 a great way to learn how to treat others and how to act around other people when you don't know how to act. If you act differently around people that you've never been around before, then you could feel embarassed or maybe even get @CAPS4. Movies can help people learn about the real world by seeing how to do those type of things as we get older. Same goes with the magazines, they also help people see what not to do or to help them understand the consequences of something that shouldn't be done. Knowing what to do from a magazine could possible save your life or perhaps maybe even someone elses life.     I don't understand why some libraries would want to banned certain materials to help people understand the things that happen in someone elses life and to help them not make the same mistakes as that person once did.")

