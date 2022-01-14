import string
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import numpy as np
from numpy import load
from viterbi import *
from utils import *
import pickle
import argparse
from MaximumMatchingSegmentation import maximumMatching, getDictionary

def load_model(path_A, path_B, path_tag):
    A = load(path_A)
    B = load(path_B)
    with open(path_tag, 'rb') as fp:
        tag_counts = pickle.load(fp)
    return A,B,tag_counts

def test(sentence, path_dict, path_A, path_B, path_tag):

    vocab = read_dict(path_dict)
    A, B, tag_counts = load_model(path_A, path_B, path_tag)
    states = sorted(tag_counts.keys())
    prep = maximumMatching(sentence, 4).split()
    best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)
    best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)
    pred = viterbi_backward(best_probs, best_paths, prep, states)
    pred = [x if (x!='--s--') else 'CH' for x in pred]
    return prep, pred

# Punctuation characters
punct = set(string.punctuation)

def get_word_tag(line, vocab): 
    # Nếu là khoảng trống thì là vị trí bắt đầu
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab: 
            # Xử lý các từ không có trong từ điển là unknown
            word = assign_unk(word)
        return word, tag
    return None 


def preprocess(vocab, data_fp):
    """
    Preprocess data
    """
    data = []
    file = open(data_fp, encoding='utf-8').readlines()
    
    for index, word in enumerate(file):
        if not word.split():
            word = '--n--'
            data.append(word)
            continue
        elif word.strip() not in vocab:
            word = '--unk--'
            data.append(word)
            continue
        data.append(word.strip())
    return word , data


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    return "--unk--"

def read_data(path):
    """
    Read data with tag
    """
    # load in the training corpus
    with open(path, 'r', encoding='utf-8') as f:
        training_corpus = f.readlines()
    return training_corpus

def read_dict(path):
    """
     Read the vocabulary data, split by each line of text, and save the list
    """
    with open(path, 'r', encoding='utf-8') as f:
        voc_l = f.read().split('\n')
    # vocab: dictionary that has the index of the corresponding words
    vocab = {} 
    voc_l = list(filter(lambda a: a != '', voc_l))
    voc_l = list(set(voc_l))
    # Get the index of the corresponding words. 
    for i, word in enumerate(sorted(voc_l)): 
        vocab[word] = i       
    return vocab    

# GRADED FUNCTION: create_dictionaries
def create_dictionaries(training_corpus, vocab):
    """
    Input: 
        training_corpus: Dữ liệu training với nhãn tag.
        vocab: bộ từ điển với chứa các từ có trong bộ dữ liệu training
    Output: 
        emission_counts: Từ điển với key dưới dạng (nhãn, từ) hay (tag, word) và giá trị của xuất hiện của nó
        transition_counts: Từ điển với key dưới dạng bigram hay (prev_tag, tag) và giá trị của xuất hiện của nó
        tag_counts: Từ điển chứa các nhãn và số lần xuất hiện của nó
    """
    
    # Khởi tạo các từ điển 
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    # Khởi tạo "prev_tag" 
    prev_tag = '--s--' 
    
    i = 0 
    for word_tag in training_corpus:
        i += 1
        # Tách nhãn và từ
        word, tag = get_word_tag(word_tag, vocab)
        # tăng giá trị chuyển trạng thái giữa các nhãn tag lên 1
        transition_counts[(prev_tag, tag)] += 1
        # tăng giá trị quan sát được lên 1
        emission_counts[(tag, word)] += 1
        # tăng giá trị nhãn xuất hiện lên 1
        tag_counts[(tag)] += 1
        # gán lại giá trị nhãn trước là nhãn hiện tại
        prev_tag = tag
        
    return emission_counts, transition_counts, tag_counts

def predict_pos(prep, y, emission_counts, vocab, states):
    '''
    Input: 
        prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
        y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags for this assignment
    Output: 
        accuracy: Number of times you classified a word correctly
    '''
    
    # Initialize the number of correct predictions to zero
    num_correct = 0
    
    # Get the (tag, word) tuples, stored as a set
    all_words = set(emission_counts.keys())
    
    # Get the number of (word, POS) tuples in the corpus 'y'
    total = len(y)
    for word, y_tup in zip(prep, y): 

        # Split the (word, POS) string into a list of two items
        y_tup_l = y_tup.split()
        
        # Verify that y_tup contain both word and POS
        if len(y_tup_l) == 2:
            
            # Set the true POS label for this word
            true_label = y_tup_l[1]

        else:
            # If the y_tup didn't contain word and POS, go to next word
            continue
    
        count_final = 0
        pos_final = ''
        
        # If the word is in the vocabulary...
        if word in vocab:
            for pos in states:

            ### START CODE HERE (USE the given hints) ###
            
                # define the key as the tuple containing the POS and word
                pos_word_tup_key = (pos, word)
                # check if the (pos, word) key exists in the emission_counts dictionary
                if pos_word_tup_key in emission_counts:
                    # get the emission count of the (pos,word) tuple 
                    count = emission_counts.get(pos_word_tup_key)
                    # keep track of the POS with the largest count
                    if count > count_final:
                        # update the final count (largest count)
                        count_final = count
                        # update the final POS
                        pos_final = pos
                    
            # If the final POS (with the largest count) matches the true POS:
            if pos_final == true_label: 
                
                # Update the number of correct predictions
                num_correct += 1
            
    ### END CODE HERE ###
    accuracy = num_correct / total
    
    return accuracy

def compute_accuracy(pred, y):
    '''
    Input: 
        pred: a list of the predicted parts-of-speech 
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output: 
        
    '''
    num_correct = 0
    total = 0
    
    # Zip together the prediction and the labels
    for prediction, y in zip(pred, y):
        # Split the label into the word and the POS tag
        word_tag_tuple = y.replace("\n", '').split("\t")
        
        # Check that there is actually a word and a tag
        # no more and no less than 2 items
        if len(word_tag_tuple) != 2: # complete this line
            continue 

        # store the word and tag separately
        word, tag = word_tag_tuple
        
        # Check if the POS tag label matches the prediction
        if prediction == tag: # complete this line
            
            # count the number of times that the prediction
            # and label match
            num_correct += 1
            
        # keep track of the total number of examples (that have valid labels)
        total += 1
        
    return num_correct/total