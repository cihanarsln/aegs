import pandas as pd
import numpy as np
import preprocessing
import vectorization

from pandas import DataFrame


def find_topic(essay):



    print("finish")

'''
    Pre_calculated values
    This part of the code steals too much time at runtime
    So store data csv files its faster
'''
def essays_to_csv():
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

    computer_tfidf = vectorization.find_word_vector_v2(computers)
    library_tfidf = vectorization.find_word_vector_v2(libraries)
    cyclist_tfidf = vectorization.find_word_vector_v2(cyclists)
    history_tfidf = vectorization.find_word_vector_v2(histories)
    memoir_tfidf = vectorization.find_word_vector_v2(memoirs)
    mooring_tfidf = vectorization.find_word_vector_v2(moorings)
    patience_tfidf = vectorization.find_word_vector_v2(patiences)
    laughter_tfidf = vectorization.find_word_vector_v2(laughters)

    dataset = []
    create_dataset(dataset, computers, libraries, cyclists, histories, memoirs, moorings, patiences, laughters, computer_tfidf, 'computer')
    create_dataset(dataset, libraries, computers, cyclists, histories, memoirs, moorings, patiences, laughters, library_tfidf, 'library')
    create_dataset(dataset, cyclists, computers, libraries, histories, memoirs, moorings, patiences, laughters, cyclist_tfidf, 'cyclist')
    create_dataset(dataset, histories, computers, libraries, cyclists, memoirs, moorings, patiences, laughters, history_tfidf, 'history')
    create_dataset(dataset, memoirs, computers, libraries, cyclists, histories, moorings, patiences, laughters, memoir_tfidf, 'memoir')
    create_dataset(dataset, moorings, computers, libraries, cyclists, histories, memoirs, patiences, laughters, mooring_tfidf, 'mooring')
    create_dataset(dataset, patiences, computers, libraries, cyclists, histories, memoirs, moorings, laughters, patience_tfidf, 'patience')
    create_dataset(dataset, laughters, computers, libraries, cyclists, histories, memoirs, moorings, patiences, laughter_tfidf, 'laughter')

    df = DataFrame(dataset, columns=['computer', 'library', 'cyclist', 'history', 'memoir', 'mooring', 'patience', 'laughter', 'label'])
    df.to_csv('topic.csv', index=False)

def create_dataset(dataset, own, other1, other2, other3, other4, other5, other6, other7, own_tfidf, label):
    for i in range(len(own)):
        values = [0, 0, 0, 0, 0, 0, 0, 0, ""]
        values[0] = own_tfidf[i]
        with_library = np.append(other1, own[i])
        values[1] = vectorization.find_word_vector_v2(with_library)[-1]
        with_cyclist = np.append(other2, own[i])
        values[2] = vectorization.find_word_vector_v2(with_cyclist)[-1]
        with_history = np.append(other3, own[i])
        values[3] = vectorization.find_word_vector_v2(with_history)[-1]
        with_memoir = np.append(other4, own[i])
        values[4] = vectorization.find_word_vector_v2(with_memoir)[-1]
        with_mooring = np.append(other5, own[i])
        values[5] = vectorization.find_word_vector_v2(with_mooring)[-1]
        with_patience = np.append(other6, own[i])
        values[6] = vectorization.find_word_vector_v2(with_patience)[-1]
        with_laughter = np.append(other7, own[i])
        values[7] = vectorization.find_word_vector_v2(with_laughter)[-1]
        values[8] = label
        dataset.append(values)
