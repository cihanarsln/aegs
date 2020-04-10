from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def find_word_vector(essays):
    sentences = []
    for i in range(len(essays)):
        temp = ' '
        sentences.append(temp.join(essays[i]))
    x = vectorizer.fit_transform(sentences)
    idf = []
    for i in range(x.shape[0]):
        temp = sum(x[i, :].data)
        idf.append(temp)
    return idf
