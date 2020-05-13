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

# This part of the code to create result.csv it is ready for use random forest
# Only tf-idf values must change
"""
# Read first 1783 essays and scores
essays_scores = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
essays_scores = essays_scores.iloc[:1783, :]
essays = essays_scores['essay']
scores1 = essays_scores['rater1_domain1']
scores2 = essays_scores['rater2_domain1']

---
    # Create dataset
    dataset: feature array
    [0]: sentence count
    [1-2]: true/false spelled word count
    [3-21]: postag counts
    [22]: word count with out stopwords
    [23]: tf-idf
    [24]: mean score
    [25]: essay without stopwords
---
sentence_counts = sentence.find_counts(essays)
words_without_stopwords = preprocessing.remove_stopwords(essays)
tf_idf_values = vectorization.find_word_vector(words_without_stopwords)
dataset = util.combine_lists(sentence_counts, words_without_stopwords, tf_idf_values, scores1, scores2)

# Create dataframe for Random Forest Algorithm
df = DataFrame(dataset, columns=['sentence_count', 'english_word', 'non_english_word', 'CC', 'DT-PDT', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'RB2', 'RBR', 'RBS', 'VB', 'VBD-VBN', 'VBG', 'VBP-VBZ', 'other_tags', 'word_count', 'td_idf', 'score', 'essay_wo_stopwords'])

df.to_csv('result.csv', index=False)

"""

def aegs(essay):

    a = topic.find_topic(essay)

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
    print("dsfd")
    '''
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('--------------------------------')
    for i in range(len(y_pred)):
        print(y_test[i], y_pred[i])

    print("finish")
    '''

aegs("I think we can all agree that computer usage is a very controversal issue. In my opinion, I believe that computers have a negative effect on people. For instance, it's not safe and children can get into all sorts of things on the internet. Also, people spend too much time in front the computer now a days, @CAPS1, its a major distraction and also a negetive effect on kids. school work. It's now or never! Do we dicide that computers have a negetive effect? You decide! Isn't every parents biggest concern the safety of their children? When on the internet, kids are capable of accessing anything and everything. Sometimes kids don't even look for bad things, they just pop up. Would you want your child veiwing things that you have no control over? Also, websites like @CAPS2.com one one of the greatest concerns when it comes to internet safety. Although you are supposed to be at least @NUM1 to have a @CAPS2, most kids lie about their age. Did you know that @NUM2 out of @NUM3 @CAPS2 users lie about their age? And it's not always a @NUM4 year old saying they are @NUM1, it could be a @NUM6 year old saying they're @NUM7! Not only do people lie about their age, they lie about who they are. Is this the kind of internet exposer you want for your children? Put a stop to this right now! More than @PERCENT1 of @CAPS3 are overweight and unhealthy. This is another negetive effect computers have on people. It's a gorgeous @DATE1 day. Bright blue skies, cotton candy cloulds, the sun is shining, and there's a nice warm breece. Perfect day to go out and get active, right? Wrong! None people would @CAPS5 be inside on the computer. Instead of going for a walk, people would @CAPS5 spend hours on facebook. This is a serious concern to our health. People don't exercise enough as it is, and then when you add computers, people will never get active! Instead of playing video games onlin, people need to be reminded that turning off the computer and playing a fun beighborhood game of baseball is just as fun and much more beneficial. This is just one step @CAPS3 need to take to get a healthier lifestyle. Wouldn't you agree? Did you know that kids that spend more time on computer are more likely to do poorly in school? Surely, if nothing else will convince you of the negetive effects of a computer this will @CAPS5 than coming home and doing homework, more time is spent in front of the computer. As a student, I will admit that the computer is a very tempting distraction and can easily pull a student away from their studies. You can't expect a child to make the right decision and tell their they have to go because they need to study. So you do! Take action now, or your child will definately suffer. The time has come to decide. Do you believe Computers have a negative effect on people? It's clear that the computer is not safe. Not to mention, too much time is spent on the computer instead of being active. Most importantly, computers will negetively affect children's grades. Don't wait another minute! Let's agree and do something about this!")




