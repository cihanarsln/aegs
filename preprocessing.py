import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

'''
    Remove characters without a-zA-Z0-9@
    Lowercase
    Tokenize
    Remove stop words
'''
def remove_stopwords(essays):
    temp = []
    for i in range(len(essays)):
        essay = re.sub('[^a-zA-Z0-9@]', ' ', essays['essay'][i])
        essay = essay.lower()
        essay = nltk.word_tokenize(essay)
        essay = [wnl.lemmatize(word) for word in essay if not word in set(stopwords.words('english'))]
        temp.append(essay)
    return temp