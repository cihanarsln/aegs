import pandas as pd
import numpy as np
import vectorization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def find_topic(essay):

    # essays_to_csv()
    topic = pd.read_csv('topic.csv', encoding="ISO-8859-1")

    # Split train and test sets
    X = topic.iloc[:5309, :6].values
    y = topic.iloc[:5309, 7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    essay_data = find_essay_values(essay)

    # Scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    essay_test = sc.transform(essay_data)

    # Classifier
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    model.fit(X_train, y_train)
    essay_pred = model.predict(essay_test)

    return essay_pred[0]

def find_essay_values(essay):

    temp = ' '.join(essay[0])

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

    data = [0, 0, 0, 0, 0, 0]
    data[0] = find_score(computers, temp)[0]
    data[1] = find_score(libraries, temp)[0]
    data[2] = find_score(cyclists, temp)[0]
    data[3] = find_score(histories, temp)[0]
    data[4] = find_score(memoirs, temp)[0]
    data[5] = find_score(moorings, temp)[0]

    return [data]

def find_score(essay, essays):
    with_essay = np.append(essays, essay)
    res = vectorization.find_word_vector_v2(with_essay)[0]
    return res

# Confusion matrix
def visualize():

    topic = pd.read_csv('topic.csv', encoding="ISO-8859-1")

    # Split train and test sets
    X = topic.iloc[:5310, :6].values
    y = topic.iloc[:5310, 7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Classifier
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    cm = confusion_matrix(y_test, y_pred)
    print('---------- Topic Scores ----------')
    print(model.score(X_test, y_test))
    print(cm)