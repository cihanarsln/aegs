import numpy as np
import enchant

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

enc = enchant.Dict("en_US")

def find_sentence_counts(essays):
    sentence_counts = []
    for i in range(len(essays)):
        essay = essays[i]
        sentence_count = find_sentence_count(essay)
        sentence_counts.append(sentence_count)
    return sentence_counts

# sentence_separators = [dot, question mark, exclamation mark, colon]
def find_sentence_count(essay):
    sentence_separators = [0, 0, 0, 0]
    sentence_separators[0] = essay.count('.')
    sentence_separators[1] = essay.count('?')
    sentence_separators[2] = essay.count('!')
    sentence_separators[3] = essay.count(':')
    return sum(sentence_separators)

def find_word_counts(essays):
    word_counts = []
    for i in range(len(essays)):
        counts = find_word_count(essays[i])
        word_counts.append(counts)
    return word_counts

# find english and non-english word count and postag counts
def find_word_count(essay):
    words_counts = np.zeros([11, 1], int)
    eng = 0
    non_eng = 0
    labeled = find_labeled_word_count(essay)
    for i in range(len(essay)):
        a = essay[i]
        if enc.check(essay[i]):
            eng = eng + 1
            tag = find_postag(essay[i])
            tag = str(tag[0][1])
            if tag is 'CC': words_counts[2] = words_counts[2] + 1
            elif tag is 'CD': words_counts[3] = words_counts[3] + 1
            elif tag is 'DT': words_counts[4] = words_counts[4] + 1
            elif tag is 'IN': words_counts[5] = words_counts[5] + 1
            elif any(tag == j for j in ('JJ', 'JJR', 'JJS')): words_counts[6] = words_counts[6] + 1
            elif any(tag == j for j in ('NN', 'NNP', 'NNS', 'NNPS')): words_counts[7] = words_counts[7] + 1
            elif any(tag == j for j in ('RB', 'RBR', 'RBS')): words_counts[8] = words_counts[8] + 1
            elif any(tag == j for j in ('VB', 'VBD', 'VBG', 'VBP', 'VBZ')): words_counts[9] = words_counts[9] + 1
            else: words_counts[10] = words_counts[10] + 1
        else: non_eng = non_eng + 1
    non_eng = round((non_eng - (labeled * 2)) * 2 / 3)
    words_counts[0] = eng
    words_counts[1] = non_eng
    return words_counts.transpose()


def find_labeled_word_count(essay):
    count = 0
    for i in range(len(essay)):
        if essay[i] == "@": count = count+1
    return count

def find_postag(word):
    token = word_tokenize(word)
    tag = pos_tag(token)
    return tag