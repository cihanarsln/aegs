import numpy as np
import pandas as pd
import nltk
import re
import enchant

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
wnl = WordNetLemmatizer()
cv = CountVectorizer(max_features=2000)
enc = enchant.Dict("en_US")

# Read all essays and scores
essays = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
essays = essays.iloc[:1783, :]

sentence_counts = []
word_counts = []
english_words_counts = []
non_english_words_counts = []
pos_tag_counts = []

# parameters | e: essay , ss: sentence_separators
def find_sentenceSeparatorsCount(e):
    # sentence_separators = [dot, question mark, exclamation mark, colon]
    sentence_separators = [0, 0, 0, 0]
    sentence_separators[0] = e.count('.')
    sentence_separators[1] = e.count('?')
    sentence_separators[2] = e.count('!')
    sentence_separators[3] = e.count(':')
    return sum(sentence_separators)

# 1) Sentence count in essay
def find_sentenceCount():
    for i in range(len(essays)):
        essay = essays['essay'][i]
        sentence_count = find_sentenceSeparatorsCount(essay)
        sentence_counts.append(sentence_count)
    return sentence_counts

def find_countLabeledWord(essay):
    count = 0
    for i in range(len(essay)):
        if essay[i] == "@": count = count+1
    return count


def find_postags(word, pos_tags):
    pass

def find_nonEnglishWordCount(essay):
    pos_tags = np.zeros([1, 13], int)
    eng = 0
    non_eng = 0
    labeled = find_countLabeledWord(essay)
    for i in range(len(essay)):
        if enc.check(essay[i]):
            eng = eng + 1
            find_postags(essay[i], pos_tags)
        else: non_eng = non_eng + 1
    english_words_counts.append(eng)
    non_eng = round((non_eng-(labeled*2))*2/3)
    non_english_words_counts.append(non_eng)

# 2) Word count in essay without stop words
def find_wordCount():
    for i in range(len(essays)):
        essay = re.sub('[^a-zA-Z0-9@]', ' ', essays['essay'][i])
        essay = essay.lower()
        essay = nltk.word_tokenize(essay)
        essay = [wnl.lemmatize(word) for word in essay if not word in set(stopwords.words('english'))]
        word_counts.append(len(essay))
        find_nonEnglishWordCount(essay)


def find_countVector():
    x = cv.fit_transform(essays['essay']).toarray()
    return x


a = pos_tag(word_tokenize("Every Saturday Daniel and his family go to the beach."))
#find_sentenceCount()
#find_wordCount()
#a = find_countVector()
print(enc.check(""))


