import numpy as np
from collections import defaultdict
from numpy import asarray
from numpy import save


class HiddenMarkovModel():

    def __init__(self, smooth, tag_counts, transition_counts, emission_counts, vocab):
        super(HiddenMarkovModel, self).__init__()
        self.smooth = smooth
        self.tag_counts = tag_counts
        self.transition_counts = transition_counts
        self.emission_counts = emission_counts
        self.vocab = list(vocab.keys())
        
    # GRADED FUNCTION: create_transition_matrix
    def create_transition_matrix(self):
        ''' 
        Input: 
            alpha: number used for smoothing
            tag_counts: a dictionary mapping each tag to its respective count
            transition_counts: transition count for the previous word and tag
        Output:
            A: matrix of dimension (num_tags,num_tags)
        '''
        # Get a sorted list of unique POS tags
        all_tags = sorted(self.tag_counts.keys())
        
        # Count the number of unique POS tags
        num_tags = len(all_tags)
        
        # Initialize the transition matrix 'A'
        A = np.zeros((num_tags,num_tags))
        
        # Get the unique transition tuples (previous POS, current POS)
        trans_keys = set(self.transition_counts.keys())
        
        # Go through each row of the transition matrix A
        for i in range(num_tags):
            
            # Go through each column of the transition matrix A
            for j in range(num_tags):

                # Initialize the count of the (prev POS, current POS) to zero
                count = 0
            
                # Define the tuple (prev POS, current POS)
                # Get the tag at position i and tag at position j (from the all_tags list)
                key = (all_tags[i], all_tags[j])

                # Check if the (prev POS, current POS) tuple 
                # exists in the transition counts dictionaory
                if key in trans_keys: #complete this line
                    
                    # Get count from the transition_counts dictionary 
                    # for the (prev POS, current POS) tuple
                    count = self.transition_counts.get(key)
                    
                # Get the count of the previous tag (index position i) from tag_counts
                count_prev_tag = self.tag_counts.get(all_tags[i])
                
                # Apply smoothing using count of the tuple, alpha, 
                # count of previous tag, alpha, and number of total tags
                A[i,j] = (count + self.smooth) / (count_prev_tag + self.smooth * num_tags)
        return A
    
    # GRADED FUNCTION: create_emission_matrix

    def create_emission_matrix(self):
        '''
        Input: 
            alpha: tuning parameter used in smoothing 
            tag_counts: a dictionary mapping each tag to its respective count
            emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
            vocab: a dictionary where keys are words in vocabulary and value is an index
        Output:
            B: a matrix of dimension (num_tags, len(vocab))
        '''
        
        # get the number of POS tag
        num_tags = len(self.tag_counts)
        
        # Get a list of all POS tags
        all_tags = sorted(self.tag_counts.keys())
        
        # Get the total number of unique words in the vocabulary
        num_words = len(self.vocab)
        
        # Initialize the emission matrix B with places for
        # tags in the rows and words in the columns
        B = np.zeros((num_tags, num_words))
        
        # Get a set of all (POS, word) tuples 
        # from the keys of the emission_counts dictionary
        emis_keys = set(list(self.emission_counts.keys()))
        
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        
        # Go through each row (POS tags)
        for i in range(num_tags): # complete this line
            
            # Go through each column (words)
            for j in range(num_words): # complete this line

                # Initialize the emission count for the (POS tag, word) to zero
                count = 0
                        
                # Define the (POS tag, word) tuple for this row and column
                key = (all_tags[i], self.vocab[j])

                # check if the (POS tag, word) tuple exists as a key in emission counts
                if key in emis_keys: # complete this line
            
                    # Get the count of (POS tag, word) from the emission_counts d
                    count = self.emission_counts.get(key)
                    
                # Get the count of the POS tag
                count_tag = self.tag_counts.get(all_tags[i])
                    
                # Apply smoothing and store the smoothed value 
                # into the emission matrix B for this row and column
                B[i,j] = (count + self.smooth) / (count_tag + self.smooth * num_words)

        ### END CODE HERE ###
        return B

