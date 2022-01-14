from utils import *
from HiddenMarkovModel import *
from numpy import asarray
from numpy import save
import pickle
import argparse

def argument():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_train", "--path_train", help='data training',default='data_tag/train.txt')
    parser.add_argument("--path_dict", "--path_dict", help='dictionary', default='WordSegmentation/vocabs.txt')
    return parser

def train(path_train, path_dict):

    training_corpus = read_data(path_train)
    vocab = read_dict(path_dict)

    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
    # get all the POS states
    states = sorted(tag_counts.keys())
    print(f"Number of POS tags (number of 'states'): {len(states)}")
    print("View these POS tags (states)")
    print(states)
 

    '''
        if most_frequent_tag == True:
            accuracy_predict_pos = predict_pos(prep, testing_corpus, emission_counts, vocab, states)
            print(f"Accuracy of prediction using predict_pos is {accuracy_predict_pos:.4f}")
    '''

    model = HiddenMarkovModel(smooth=1, tag_counts = tag_counts, transition_counts = transition_counts, emission_counts = emission_counts, vocab= vocab)
    A = model.create_transition_matrix()
    B = model.create_emission_matrix()

    # Save transition matrix A
    trasition = asarray(A)
    # save to npy file
    save('./save_dir/transition_matrix.npy', trasition)

    # Save emission matrix B
    trasition = asarray(B)
    # save to npy file
    save('./save_dir/emission_matrix.npy', trasition)

    with open('./save_dir/tag.pickle', 'wb') as fp:
        pickle.dump(tag_counts, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return A, B

def main():
    parser = argument()
    args = parser.parse_args()
    train(args.path_train, args.path_dict)

if __name__ == "__main__":
    main()
