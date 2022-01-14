import pickle,ast

def getDictionary():

    list_Words = []
    with open('dictionary/mydict.txt', 'r', encoding='utf-8') as f:
        text = f.readlines()
    for i in text:
        list_Words.append(i.replace("\n",""))
    bi_grams = list(load_n_grams('dictionary/bi_grams.txt'))
    tri_grams = list(load_n_grams('dictionary/tri_grams.txt'))

    for i in bi_grams:
        list_Words.append(i.replace(" ","_"))

    for i in tri_grams:
        list_Words.append(i.replace(" ","_"))
    print((len(list_Words)))
    return list_Words

def maximumMatching(sentence, maxlen):

    VNdictionary = getDictionary()
    my_list = []
    list_Words = sentence.split()
    len_now = len(list_Words)
    i = 0
    maxl = maxlen
    while i < len_now:
        word = '_'.join(list_Words[i:maxlen + i])
        while word not in VNdictionary:
            if maxl > 1:
                word = '_'.join(list_Words[i:i + maxl])
            elif maxl == 1:
                word = list_Words[i]
            elif maxl == 0:
                break
            maxl -= 1
        if maxl > 0:
            i += maxl
        i += 1
        my_list.append(word)
        maxl = min(len_now - i, maxlen)

    line = ' '.join(my_list)
    return line

def WordSegmentation(path='raw_sentences.txt', maxlen=4):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.readlines()

    with open('tokenizer/MM_tokenizer.txt', 'w', encoding='utf-8') as f:
        for i in text:
            sample = maximumMatching(i,maxlen).split()
            for j in sample:
                f.write('{}\n'.format(j))
            f.write('.\n')
            f.write('\n')

def load_n_grams(path):
    with open(path, encoding='utf8') as f:
        words = f.read()
        words = ast.literal_eval(words)
    return words
def WordSegmentation2(path='raw_sentences.txt', maxlen=4):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.readlines()

    with open('result.txt', 'w', encoding='utf-8') as f:
        for i in text:
            f.write('{}\n'.format(maximumMatching(i,maxlen)))
