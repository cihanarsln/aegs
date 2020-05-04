import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def remove_stopwords(essays):
    temp = []
    for i in range(len(essays)):
        essay = re.sub('[^a-zA-Z0-9@]', ' ', essays[i])
        essay = essay.lower()
        essay = nltk.word_tokenize(essay)
        essay = [wnl.lemmatize(word) for word in essay if not word in set(stopwords.words('english'))]
        temp.append(essay)
    return temp

"""
def find_word_count_after_removing_stopwords(essay):
    temp = essay
    temp = re.sub('[^a-zA-Z0-9@]', ' ', temp)
    temp = temp.lower()
    temp = nltk.word_tokenize(temp)
    temp = [wnl.lemmatize(word) for word in temp if not word in set(stopwords.words('english'))]
    return len(temp)
"""

def remove_unnecessary_characters(essay):
    temp = re.sub('[^a-zA-Z0-9\'@]', ' ', essay)
    return temp

def remove_unnecessary_characters_v2(essay):
    temp = re.sub('[^a-zA-Z]', ' ', essay)
    return temp