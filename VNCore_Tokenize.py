from vncorenlp import VnCoreNLP
client = VnCoreNLP(address="http://127.0.0.1", port=9001)
with open('Word-Tagging/WordSegmentation/raw_sentences.txt','r',encoding ='utf-8') as f:
    sentences = f.readlines()
with open('vncore_tokens.txt', 'w', encoding='utf-8') as f:
    vncore_sentences = []
    for sentence in sentences:
        word_list = client.tokenize(sentence)[0]
        f.write(' '.join(word_list))
        if sentence != sentences[-1]: f.write('\n')
        
