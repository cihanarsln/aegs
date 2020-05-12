import pandas as pd
import numpy as np
import preprocessing
import vectorization

from pandas import DataFrame


def find_topic(essay):



    print("asdfs")

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

    essays_computer_without_stopwords = preprocessing.remove_stopwords_v2(essays_computer)
    essays_library_without_stopwords = preprocessing.remove_stopwords_v2(essays_library)
    essays_cyclist_without_stopwords = preprocessing.remove_stopwords_v2(essays_cyclist)
    essays_history_without_stopwords = preprocessing.remove_stopwords_v2(essays_history)
    essays_memoir_without_stopwords = preprocessing.remove_stopwords_v2(essays_memoir)
    essays_mooring_without_stopwords = preprocessing.remove_stopwords_v2(essays_mooring)
    essays_patience_without_stopwords = preprocessing.remove_stopwords_v2(essays_patience)
    essays_laughter_without_stopwords = preprocessing.remove_stopwords_v2(essays_laughter)
    '''
    df = DataFrame(essays_computer_without_stopwords, columns=['essay'])
    df.to_csv('essay_computer.csv', index=False)
    df = DataFrame(essays_library_without_stopwords, columns=['essay'])
    df.to_csv('essay_library.csv', index=False)
    df = DataFrame(essays_cyclist_without_stopwords, columns=['essay'])
    df.to_csv('essay_cyclist.csv', index=False)
    df = DataFrame(essays_history_without_stopwords, columns=['essay'])
    df.to_csv('essay_history.csv', index=False)
    df = DataFrame(essays_memoir_without_stopwords, columns=['essay'])
    df.to_csv('essay_memoir.csv', index=False)
    df = DataFrame(essays_mooring_without_stopwords, columns=['essay'])
    df.to_csv('essay_mooring.csv', index=False)
    df = DataFrame(essays_patience_without_stopwords, columns=['essay'])
    df.to_csv('essay_patience.csv', index=False)
    df = DataFrame(essays_laughter_without_stopwords, columns=['essay'])
    df.to_csv('essay_laughter.csv', index=False)
    '''

    computer_tf_idf_values = vectorization.find_word_vector_v2(essays_computer_without_stopwords)
    library_tf_idf_values = vectorization.find_word_vector_v2(essays_library_without_stopwords)
    cyclist_tf_idf_values = vectorization.find_word_vector_v2(essays_cyclist_without_stopwords)
    history_tf_idf_values = vectorization.find_word_vector_v2(essays_history_without_stopwords)
    memoir_tf_idf_values = vectorization.find_word_vector_v2(essays_memoir_without_stopwords)
    mooring_tf_idf_values = vectorization.find_word_vector_v2(essays_mooring_without_stopwords)
    patience_tf_idf_values = vectorization.find_word_vector_v2(essays_patience_without_stopwords)
    laughter_tf_idf_values = vectorization.find_word_vector_v2(essays_laughter_without_stopwords)

    dataset = []
    for i in range(len(essays_computer_without_stopwords)):
        values = [0, 0, 0, 0, 0, 0, 0, 0, ""]
        values[0] = computer_tf_idf_values[i]
        with_library = np.append(essays_library_without_stopwords, essays_computer_without_stopwords[i])
        values[1] = vectorization.find_word_vector_v2(with_library)[-1]
        with_cyclist = np.append(essays_cyclist_without_stopwords, essays_computer_without_stopwords[i])
        values[2] = vectorization.find_word_vector_v2(with_cyclist)[-1]
        with_history = np.append(essays_history_without_stopwords, essays_computer_without_stopwords[i])
        values[3] = vectorization.find_word_vector_v2(with_history)[-1]
        with_memoir = np.append(essays_memoir_without_stopwords, essays_computer_without_stopwords[i])
        values[4] = vectorization.find_word_vector_v2(with_memoir)[-1]
        with_mooring = np.append(essays_mooring_without_stopwords, essays_computer_without_stopwords[i])
        values[5] = vectorization.find_word_vector_v2(with_mooring)[-1]
        with_patience = np.append(essays_patience_without_stopwords, essays_computer_without_stopwords[i])
        values[6] = vectorization.find_word_vector_v2(with_patience)[-1]
        with_laughter = np.append(essays_laughter_without_stopwords, essays_computer_without_stopwords[i])
        values[7] = vectorization.find_word_vector_v2(with_laughter)[-1]
        values[8] = "computer"
        dataset.append(values)