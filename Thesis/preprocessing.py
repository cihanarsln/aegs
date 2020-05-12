import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

def remove_stopwords_v2(essays):
    temp = []
    for i in range(len(essays)):
        essay = re.sub('[^a-zA-Z0-9]', ' ', essays[i])
        essay = essay.lower()
        essay = nltk.word_tokenize(essay)
        essay = [wnl.lemmatize(word) for word in essay if not word in set(stopwords.words('english'))]
        essay = ' '.join(essay)
        temp.append(essay)
    return temp

def remove_unnecessary_characters(essay):
    temp = re.sub('[^a-zA-Z0-9\'@]', ' ', essay)
    return temp

def remove_unnecessary_characters_v2(essay):
    temp = re.sub('[^a-zA-Z]', ' ', essay)
    return temp

def remove_labeled_words(essay):
    words = word_tokenize(essay)
    indexes = []
    for i in range(len(words)):
        word = str(words[i])
        if word.startswith('@'):
            indexes.append(i)
    for i in range(len(indexes)):
        index = indexes[i] - (i*2)
        del words[index:index+2]
    temp = ' '.join(words)
    res = [temp, len(indexes)]
    return res