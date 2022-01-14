import pickle

def getDictionary(pathDictionary='VietNamDictionary.pkl'):

    list_Words = pickle.load(open(pathDictionary, 'rb'))

    with open('diadanh.txt','r', encoding='utf-8') as f:
        VNlocations = f.readlines()
        for i in VNlocations:
            i = i[0:-1]
            a = i.split()
            list_Words.append('_'.join(a))

    with open('ten.txt','r', encoding='utf-8') as f:
        VNnames = f.readlines()
        for i in VNnames:
            i = i[0:-1]
            a = i.split()
            list_Words.append('_'.join(a))

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

    with open('result.txt', 'w', encoding='utf-8') as f:
        for i in text:
            f.write('{}\n'.format(maximumMatching(i,maxlen)))

