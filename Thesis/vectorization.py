from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

# Find sum of the tf-idf values for each essay
def find_word_vector(essays):
    sentences = []
    for i in range(len(essays)):
        temp = ' '
        sentences.append(temp.join(essays[i]))
    x = vectorizer.fit_transform(sentences)
    idf = []
    for i in range(x.shape[0]):
        temp = (sum(x[i, :].data), len(x[i, :].data))
        idf.append(temp)
    return idf

def find_word_vector_v2(essays):
    x = vectorizer.fit_transform(essays)
    idf = []
    for i in range(x.shape[0]):
        temp = sum(x[i, :].data)
        idf.append(temp)
    return idf
