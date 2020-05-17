import pandas as pd
import sentence
import preprocessing
import vectorization

from pandas import DataFrame

def combine_lists(sentence_counts, words_without_stopwords, tf_idf_values, scores1, scores2):
    temp = []
    for i in range(len(sentence_counts)):
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '']
        essay = ' '.join(words_without_stopwords[i])
        for j in range(len(sentence_counts[0])):
            data[j] = sentence_counts[i][j]
        data[22] = len(words_without_stopwords[i])
        data[23] = tf_idf_values[i][0]
        data[24] = tf_idf_values[i][1]
        data[25] = (scores1[i]+scores2[i])/2
        data[26] = essay
        temp.append(data)
    return temp

def create_pre_calculated_result_csv(start, end, csv_name):

    # Read first 1783 essays and scores
    essays_scores = pd.read_csv('essays_and_scores.csv', encoding="ISO-8859-1")
    essays_scores = essays_scores.iloc[start:end, :]
    essays = essays_scores['essay'].values
    scores1 = essays_scores['rater1_domain1'].values
    scores2 = essays_scores['rater2_domain1'].values

    '''
    # Create dataset
    dataset: feature
    array
    [0]: sentence
    count
    [1 - 2]: true / false
    spelled
    word
    count
    [3 - 21]: postag
    counts
    [22]: word
    count
    with out stopwords
    [23]: tf - idf
    [24]: mean
    score
    [25]: essay
    without
    stopwords
    '''
    sentence_counts = sentence.find_counts(essays)
    words_without_stopwords = preprocessing.remove_stopwords(essays)
    tf_idf_values = vectorization.find_word_vector(words_without_stopwords)
    dataset = combine_lists(sentence_counts, words_without_stopwords, tf_idf_values, scores1, scores2)

    # Create dataframe for Random Forest Algorithm
    df = DataFrame(dataset,
                   columns=['sentence_count', 'english_word', 'non_english_word', 'CC', 'DT-PDT', 'IN', 'JJ', 'JJR',
                            'JJS',
                            'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'RB2', 'RBR', 'RBS', 'VB', 'VBD-VBN', 'VBG',
                            'VBP-VBZ',
                            'other_tags', 'word_count', 'td_idf', 'score', 'essay_wo_stopwords'])

    df.to_csv(csv_name, index=False)