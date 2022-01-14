from vncorenlp import VnCoreNLP
client = VnCoreNLP(address="http://127.0.0.1", port=9001)

manual_tokens = open('tokenizer/manual_Tokenizer.txt', encoding='utf-8').readlines()
print('Số lượng từ:', len(manual_tokens))

with open('data_tag/manual_Tagging.txt', 'w', encoding='utf-8') as f:
    for word in manual_tokens:
        word = word.replace('\n', '')
        
        if '_' not in word: tag = client.pos_tag(word)
        else: tag = client.pos_tag(word.replace('_', ' '))
        
        if tag == []: f.write('\n')
        else: f.write(f'{word}\t{tag[0][0][1]}\n')
    f.write('\n')