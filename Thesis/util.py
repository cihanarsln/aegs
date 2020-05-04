def combine_lists(sentence_counts, words_without_stopwords, tf_idf_values, scores1, scores2):
    temp = []
    for i in range(len(sentence_counts)):
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '']
        essay = ' '.join(words_without_stopwords[i])
        for j in range(len(sentence_counts[0])):
            data[j] = sentence_counts[i][j]
        data[22] = len(words_without_stopwords[i])
        data[23] = tf_idf_values[i]
        data[24] = (scores1[i]+scores2[i])/2
        data[25] = essay
        temp.append(data)
    return temp