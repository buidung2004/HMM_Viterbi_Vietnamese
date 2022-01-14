import numpy as np
from numpy import load
from viterbi import *
from utils import *
import pickle
import argparse
from MaximumMatchingSegmentation import maximumMatching, getDictionary
import argparse

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dict", "--path_dict", help='dictionary', default='WordSegmentation/vocabs.txt')
    parser.add_argument("--path_B", "--path_B", help='dictionary', default= 'save_dir/emission_matrix.npy')
    parser.add_argument("--path_A", "--path_A", help='dictionary', default='save_dir/transition_matrix.npy')
    parser.add_argument("--path_tag", "--path_tag", help='dictionary', default='save_dir/tag.pickle')
    parser.add_argument("--sentence", "--sentence", help='dictionary', type=str, default='liên quan đến pháp luật .')

    return parser

def load_model(path_A, path_B, path_tag):
    A = load(path_A)
    B = load(path_B)
    with open(path_tag, 'rb') as fp:
        tag_counts = pickle.load(fp)
    return A,B,tag_counts

def test(sentence, path_dict, path_A, path_B, path_tag):

    vocab = read_dict(path_dict)
    A, B, tag_counts = load_model(path_A, path_B, path_tag)
    print(tag_counts)
    states = sorted(tag_counts.keys())
    prep = maximumMatching(sentence, 4).split()
    print(prep)
    best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)
    best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)
    pred = viterbi_backward(best_probs, best_paths, prep, states)
    pred = [x if (x!='--s--') else 'CH' for x in pred]
    print('Nhãn được gán là', pred)

def main():
    parser = argument()
    args = parser.parse_args()
    test(args.sentence, args.path_dict, args.path_A, args.path_B, args.path_tag)

if __name__ == "__main__":
    main()

    