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
