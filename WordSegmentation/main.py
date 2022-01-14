import pickle
from MaximumMatchingSegmentation import *
import argparse

def argument():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", "--path_data", help='txt file contains sentences',default='raw_sentences.txt')
    parser.add_argument("--maxlen", "--maxlen", help='longest length of words', default=4, type=int)
    return parser



def main():
    parser = argument()
    args = parser.parse_args()
    WordSegmentation(args.path_data, args.maxlen)

if __name__ == "__main__":
    main()