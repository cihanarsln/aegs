import enchant

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from spellchecker import SpellChecker

import preprocessing

enc = enchant.Dict("en_US")
spell = SpellChecker()

"""
    essay_counts: feature array
    [0]: sentence count
    [1-2]: true/false spelled word count
    [3-21]: postag counts
"""
def find_counts(essays):
    counts = []
    for i in range(len(essays)):
        essay_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        essay = essays[i]
        essay_tokenized = sent_tokenize(essay)
        sentence_count = find_sentence_count(essay_tokenized)
        postag_counts = find_postag_counts(essay)
        misspelled_word_count = find_misspelled_word_counts(essay)
        combine_lists(essay_counts, sentence_count, misspelled_word_count, postag_counts)
        counts.append(essay_counts)
    return counts


"""
    # Find sentence count in essay
    # 1 Feature
"""
def find_sentence_count(essay_tokenize):
    return len(essay_tokenize)

"""
    # Find tag counts in essay 
    # 19 Feature
    # CC: Coordinating Conjunction, [ DT: Determiner, PDT: Predeterminer ], IN: Preposition/Subordinating Conjunction
    # JJ: Adjective, JJR: Adjective, Comparative, JJS: Adjective, Superlative, MD: Modal Could
    # NN: Noun (singular), NNS: Noun (plural), NNP: Proper Noun (singular), NNPS: Proper Noun (plural)
    # RB: Adverb, RBR: Adverb (comparative), RBS: Adverb (superlative)
    # VB: Verb, [ VBD: Verb (past), VBN: Verb (past) ], VBG: Verb (present), [ VBP: Verb (sing. present, known-3d take), VBZ: Verb (3rd person sing. present takes) ]
    # [CC, DT-PDT, IN, JJ, JJR, JJS, MD, NN, NNS, NNP, NNPS, RB, RBR, RBS, VB, VBD-VBN, VBG, VBP-VBZ, other_tags] 
"""
def find_postag_counts(essay):
    postag_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp = preprocessing.remove_unnecessary_characters(essay)
    postags = pos_tag(word_tokenize(temp))
    for i in range(len(postags)):
        tag = postags[i][1]
        if tag == "CC": postag_counts[0] = postag_counts[0] + 1
        elif tag == "DT": postag_counts[1] = postag_counts[1] + 1
        elif tag == "PDT": postag_counts[1] = postag_counts[1] + 1
        elif tag == "IN": postag_counts[2] = postag_counts[2] + 1
        elif tag == "JJ": postag_counts[3] = postag_counts[3] + 1
        elif tag == "JJR": postag_counts[4] = postag_counts[4] + 1
        elif tag == "JJS": postag_counts[5] = postag_counts[5] + 1
        elif tag == "MD": postag_counts[6] = postag_counts[6] + 1
        elif tag == "NN": postag_counts[7] = postag_counts[7] + 1
        elif tag == "NNS": postag_counts[8] = postag_counts[8] + 1
        elif tag == "NNP": postag_counts[9] = postag_counts[9] + 1
        elif tag == "NNPS": postag_counts[10] = postag_counts[10] + 1
        elif tag == "RB": postag_counts[11] = postag_counts[11] + 1
        elif tag == "RBR": postag_counts[12] = postag_counts[12] + 1
        elif tag == "RBS": postag_counts[13] = postag_counts[13] + 1
        elif tag == "VB": postag_counts[14] = postag_counts[14] + 1
        elif tag == "VBD": postag_counts[15] = postag_counts[15] + 1
        elif tag == "VBN": postag_counts[15] = postag_counts[15] + 1
        elif tag == "VBG": postag_counts[16] = postag_counts[16] + 1
        elif tag == "VBP": postag_counts[17] = postag_counts[17] + 1
        elif tag == "VBZ": postag_counts[17] = postag_counts[17] + 1
        else: postag_counts[18] = postag_counts[18] + 1
    return postag_counts

"""
    # [True, False]
    # 2 Feature
"""
def find_misspelled_word_counts(essay):
    counts = [0, 0]
    temp = preprocessing.remove_unnecessary_characters_v2(essay)
    words = word_tokenize(temp)
    for i in range(len(words)):
        if enc.check(words[i]):
            counts[0] = counts[0] + 1
        else:
            counts[1] = counts[1] + 1
    return counts

def find_misspelled_word_correction(essay):
    misspelled_token = ""
    label = preprocessing.remove_labeled_words(essay)
    temp = preprocessing.remove_unnecessary_characters_v2(label[0])
    words = word_tokenize(temp)
    for i in range(len(words)):
        if not enc.check(words[i]):
            word_correction = words[i] + ":" + spell.correction(words[i])
            misspelled_token += word_correction + ","
    misspelled_token = misspelled_token[:-1]
    return misspelled_token

def combine_lists(essay_counts, sentence_count, misspelled_word_count, postag_counts):
    essay_counts[0] = sentence_count
    essay_counts[1] = misspelled_word_count[0]
    essay_counts[2] = misspelled_word_count[1]
    for i in range(len(postag_counts)):
        essay_counts[i+3] = postag_counts[i]