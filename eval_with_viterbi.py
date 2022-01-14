import numpy as np
from numpy import load
from viterbi import *
from utils import *
import pickle
import argparse

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_test", "--path_test", help='data training',default='data_tag/test.txt')
    parser.add_argument("--path_dict", "--path_dict", help='dictionary', default='WordSegmentation/vocabs.txt')
    parser.add_argument("--path_test_no_tag", "--path_test_no_tag", help='dictionary', default='data_tag/test_notag.txt')
    parser.add_argument("--path_B", "--path_B", help='dictionary', default= 'save_dir/emission_matrix.npy')
    parser.add_argument("--path_A", "--path_A", help='dictionary', default='save_dir/transition_matrix.npy')
    parser.add_argument("--path_tag", "--path_tag", help='dictionary', default='save_dir/tag.pickle')
    return parser

def load_model(path_A, path_B, path_tag):
    A = load(path_A)
    B = load(path_B)
    with open(path_tag, 'rb') as fp:
        tag_counts = pickle.load(fp)
    return A,B,tag_counts

def eval(path_dict, path_test, path_test_no_tag, path_A, path_B, path_tag):

    vocab = read_dict(path_dict)
    A, B, tag_counts = load_model(path_A, path_B, path_tag)
    states = sorted(tag_counts.keys())
    # load in the test corpus
    y = read_data(path_test)
    print(y)
    
    print("Number word of test corpus ", len(y))
      #corpus without tags, preprocessed
    _, prep = preprocess(vocab, path_test_no_tag)

    # Viterbi
    best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)
    best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)
    pred = viterbi_backward(best_probs, best_paths, prep, states)
    pred = [x if (x!='--s--') else 'CH' for x in pred]
    print(len(y), len(pred))
    with open('data_tag/HMM_Viterbi_train_predict.txt','w',encoding='utf-8') as f:
        for i in range(len(prep)):
            if i == '--n--':
                f.write('\n')
            f.write('{}\t{}\n'.format(prep[i],pred[i]))
    print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")

def main():
    parser = argument()
    args = parser.parse_args()
    eval(args.path_dict, args.path_test, args.path_test_no_tag, args.path_A, args.path_B, args.path_tag)

if __name__ == "__main__":
    main()    
