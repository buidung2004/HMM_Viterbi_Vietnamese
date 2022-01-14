from numpy.core.fromnumeric import _partition_dispatcher
from vncorenlp import VnCoreNLP
from utils import *
client = VnCoreNLP(address="http://127.0.0.1", port=9001)

def VnCoreNLP_Tag(path):
    manual_tokens = open(path, encoding='utf-8').readlines()
    print('Số lượng từ:', len(manual_tokens))

    with open('data_tag/train_VnCoreNLP.txt', 'w', encoding='utf-8') as f:
        for word in manual_tokens:
            word = word.replace('\n', '')
            
            if '_' not in word: tag = client.pos_tag(word)
            else: tag = client.pos_tag(word.replace('_', ' '))
            
            if tag == []: f.write('\n')
            else: f.write(f'{word}\t{tag[0][0][1]}\n')
        f.write('\n')

def eval(path_test, path_test_CoreNLP):

    # load in the test corpus
    y = read_data(path_test)
    print("Number word of test corpus ", len(y))
      #corpus without tags, preprocessed
    VnCore_test = open(path_test_CoreNLP, encoding='utf-8').readlines()
    pred = []
    for i in VnCore_test:
        try:
            pred.append(i.split()[1])
        except:
            pred.append('\n')
    print(len(pred))
    print(f"Accuracy of the VnCoreNLP is {compute_accuracy(pred, y):.4f}")

VnCoreNLP_Tag('data_tag/train_notag.txt')
eval('data_tag/train.txt','data_tag/train_VnCoreNLP.txt')