import pandas as pd
import numpy as np
import preprocessing
import vectorization

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def find_topic(essay, score):

    # essays_to_csv()
    topic = pd.read_csv('topic.csv', encoding="ISO-8859-1")

    # Split train and test sets
    X = topic.iloc[:5309, :7].values
    y = topic.iloc[:5309, 7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    essay_data = find_essay_values(essay, score)

    # Scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    essay_test = sc.transform(essay_data)

    # Classifier
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    model.fit(X_train, y_train)

    essay_pred = model.predict(essay_test)

    # y_pred = model.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(model.score(X_test, y_test))

    return essay_pred[0]

def find_essay_values(essay, score):
    computers = pd.read_csv('essay_computer.csv', encoding="ISO-8859-1")
    libraries = pd.read_csv('essay_library.csv', encoding="ISO-8859-1")
    cyclists = pd.read_csv('essay_cyclist.csv', encoding="ISO-8859-1")
    histories = pd.read_csv('essay_history.csv', encoding="ISO-8859-1")
    memoirs = pd.read_csv('essay_memoir.csv', encoding="ISO-8859-1")
    moorings = pd.read_csv('essay_mooring.csv', encoding="ISO-8859-1")

    computers = computers.iloc[:, :].values
    libraries = libraries.iloc[:, :].values
    cyclists = cyclists.iloc[:, :].values
    histories = histories.iloc[:, :].values
    memoirs = memoirs.iloc[:, :].values
    moorings = moorings.iloc[:, :].values

    data = [0, 0, 0, 0, 0, 0, 0]
    data[0] = find_score(computers, essay)
    data[1] = find_score(libraries, essay)
    data[2] = find_score(cyclists, essay)
    data[3] = find_score(histories, essay)
    data[4] = find_score(memoirs, essay)
    data[5] = find_score(moorings, essay)
    data[6] = score

    return [data]

'''
    Pre_calculated values
    This part of the code steals too much time at runtime
    So store data csv files its faster
'''
def essays_to_csv():
    create_essays_without_stopwords_csv()
    create_dataset()

def create_dataset():
    dataset = []

    scores = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
    scores = scores.iloc[:, 3:5].values

    computers = pd.read_csv('essay_computer.csv', encoding="ISO-8859-1")
    libraries = pd.read_csv('essay_library.csv', encoding="ISO-8859-1")
    cyclists = pd.read_csv('essay_cyclist.csv', encoding="ISO-8859-1")
    histories = pd.read_csv('essay_history.csv', encoding="ISO-8859-1")
    memoirs = pd.read_csv('essay_memoir.csv', encoding="ISO-8859-1")
    moorings = pd.read_csv('essay_mooring.csv', encoding="ISO-8859-1")

    computers = computers.iloc[:, 0].values
    libraries = libraries.iloc[:, 0].values
    cyclists = cyclists.iloc[:, 0].values
    histories = histories.iloc[:, 0].values
    memoirs = memoirs.iloc[:, 0].values
    moorings = moorings.iloc[:, 0].values

    computer_tfidf = vectorization.find_word_vector_v2(computers)
    library_tfidf = vectorization.find_word_vector_v2(libraries)
    cyclist_tfidf = vectorization.find_word_vector_v2(cyclists)
    history_tfidf = vectorization.find_word_vector_v2(histories)
    memoir_tfidf = vectorization.find_word_vector_v2(memoirs)
    mooring_tfidf = vectorization.find_word_vector_v2(moorings)

    for i in range(len(computers)):
        data = [0, 0, 0, 0, 0, 0, 0, ""]
        data[0] = computer_tfidf[i]
        data[1] = find_score(libraries, computers[i])
        data[2] = find_score(cyclists, computers[i])
        data[3] = find_score(histories, computers[i])
        data[4] = find_score(memoirs, computers[i])
        data[5] = find_score(moorings, computers[i])
        data[6] = (scores[len(dataset)][0] + scores[len(dataset)][1]) / 2
        data[7] = "computer"
        dataset.append(data)
    for i in range(len(libraries)):
        data = [0, 0, 0, 0, 0, 0, 0, ""]
        data[0] = find_score(computers, libraries[i])
        data[1] = library_tfidf[i]
        data[2] = find_score(cyclists, libraries[i])
        data[3] = find_score(histories, libraries[i])
        data[4] = find_score(memoirs, libraries[i])
        data[5] = find_score(moorings, libraries[i])
        data[6] = (scores[len(dataset)][0] + scores[len(dataset)][1]) / 2
        data[7] = "library"
        dataset.append(data)
    for i in range(len(cyclists)):
        data = [0, 0, 0, 0, 0, 0, 0, ""]
        data[0] = find_score(computers, cyclists[i])
        data[1] = find_score(libraries, cyclists[i])
        data[2] = cyclist_tfidf[i]
        data[3] = find_score(histories, cyclists[i])
        data[4] = find_score(memoirs, cyclists[i])
        data[5] = find_score(moorings, cyclists[i])
        data[6] = (scores[len(dataset)][0] + scores[len(dataset)][1]) / 2
        data[7] = "cyclist"
        dataset.append(data)
    for i in range(len(histories)):
        data = [0, 0, 0, 0, 0, 0, 0, ""]
        data[0] = find_score(computers, histories[i])
        data[1] = find_score(libraries, histories[i])
        data[2] = find_score(cyclists, histories[i])
        data[3] = history_tfidf[i]
        data[4] = find_score(memoirs, histories[i])
        data[5] = find_score(moorings, histories[i])
        data[6] = (scores[len(dataset)][0] + scores[len(dataset)][1]) / 2
        data[7] = "history"
        dataset.append(data)
    for i in range(len(memoirs)):
        data = [0, 0, 0, 0, 0, 0, 0, ""]
        data[0] = find_score(computers, memoirs[i])
        data[1] = find_score(libraries, memoirs[i])
        data[2] = find_score(cyclists, memoirs[i])
        data[3] = find_score(histories, memoirs[i])
        data[4] = memoir_tfidf[i]
        data[5] = find_score(moorings, memoirs[i])
        data[6] = (scores[len(dataset)][0] + scores[len(dataset)][1]) / 2
        data[7] = "memoir"
        dataset.append(data)
    for i in range(len(moorings)):
        data = [0, 0, 0, 0, 0, 0, 0, ""]
        data[0] = find_score(computers, moorings[i])
        data[1] = find_score(libraries, moorings[i])
        data[2] = find_score(cyclists, moorings[i])
        data[3] = find_score(histories, moorings[i])
        data[4] = find_score(memoirs, moorings[i])
        data[5] = mooring_tfidf[i]
        data[6] = (scores[len(dataset)][0] + scores[len(dataset)][1]) / 2
        data[7] = "mooring"
        dataset.append(data)

    df = DataFrame(dataset, columns=['computer', 'library', 'cyclist', 'history', 'memoir', 'mooring', 'score', 'label'])
    df.to_csv('topic.csv', index=False)



def find_score(essay, essays):
    with_essay = np.append(essays, essay)
    res = vectorization.find_word_vector_v2(with_essay)[0]
    return res


def create_essays_without_stopwords_csv():
    essays_topic = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
    essays_topic = essays_topic.iloc[:, 1:3]

    essays_computer = essays_topic.iloc[0:1783, 1].values
    essays_library = essays_topic.iloc[1783:3583, 1].values
    essays_cyclist = essays_topic.iloc[3583:5309, 1].values
    essays_history = essays_topic.iloc[5309:7081, 1].values
    essays_memoir = essays_topic.iloc[7081:8886, 1].values
    essays_mooring = essays_topic.iloc[8886:10686, 1].values
    essays_patience = essays_topic.iloc[10686:12255, 1].values
    essays_laughter = essays_topic.iloc[12255:, 1].values

    computers = preprocessing.remove_stopwords_v2(essays_computer)
    libraries = preprocessing.remove_stopwords_v2(essays_library)
    cyclists = preprocessing.remove_stopwords_v2(essays_cyclist)
    histories = preprocessing.remove_stopwords_v2(essays_history)
    memoirs = preprocessing.remove_stopwords_v2(essays_memoir)
    moorings = preprocessing.remove_stopwords_v2(essays_mooring)
    patiences = preprocessing.remove_stopwords_v2(essays_patience)
    laughters = preprocessing.remove_stopwords_v2(essays_laughter)

    df = DataFrame(computers, columns=['essay'])
    df.to_csv('essay_computer.csv', index=False)
    df = DataFrame(libraries, columns=['essay'])
    df.to_csv('essay_library.csv', index=False)
    df = DataFrame(cyclists, columns=['essay'])
    df.to_csv('essay_cyclist.csv', index=False)
    df = DataFrame(histories, columns=['essay'])
    df.to_csv('essay_history.csv', index=False)
    df = DataFrame(memoirs, columns=['essay'])
    df.to_csv('essay_memoir.csv', index=False)
    df = DataFrame(moorings, columns=['essay'])
    df.to_csv('essay_mooring.csv', index=False)
    df = DataFrame(patiences, columns=['essay'])
    df.to_csv('essay_patience.csv', index=False)
    df = DataFrame(laughters, columns=['essay'])
    df.to_csv('essay_laughter.csv', index=False)
