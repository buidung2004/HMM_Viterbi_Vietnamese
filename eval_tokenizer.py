def Token_Eva(path_token):

    precision = 0
    recall = 0

    with open(path_token, 'r', encoding='utf-8') as f:
        text = f.readlines()
        tokenize_sentences = []
        sentence = ''
        for sentence in text:
            sentence = sentence.replace('\n', ' ')
            tokenize_sentences.append(sentence)

    with open('tokenizer/manual_tokens.txt', 'r', encoding='utf-8') as f:
        text = f.readlines()
        manual_tokenize_sentences = []
        sentence = ''
        for sentence in text:
            sentence = sentence.replace('\n', ' ')
            manual_tokenize_sentences.append(sentence)

    vncore_evaluation = tokenize_evaluation(tokenize_sentences, manual_tokenize_sentences)
    return (vncore_evaluation)

def count_correct_words(pred, source, n_grams=3):
    pred_words = pred.split()
    source_words = source.split()
    
    total_true, tp = 0, 0
    total_errors, fp = 0, 0
    
    idx = 0
    while idx < len(pred_words):
        if pred_words[idx] not in source_words[idx:(idx + n_grams)]: 
            if '_' in pred_words[idx]: fp += 1
            del pred_words[idx]
            total_errors += 1
        else: idx += 1
    
    idx = 0
    while idx < len(source_words):
        if source_words[idx] not in pred_words[idx:(idx + n_grams)]: 
            del source_words[idx]
        else: idx += 1
    
    if len(pred_words) < len(source_words): words = pred_words
    else: words = source_words
    
    for idx in range (len(words)):
        if pred_words[idx] == source_words[idx]:
            if '_' in pred_words[idx]: tp += 1 
            total_true += 1
                    
    return total_true, total_errors, tp, fp

def tokenize_evaluation(pred, source, n_grams=3):
    total_true = 0
    total_errors = 0
    total_words = 0
    
    pred_tp = 0
    pred_fp = 0
    
    for pred_sentence, source_sentence in zip(pred, source):
        total_words += len(source_sentence.split())
        if pred_sentence != source_sentence:
            true, error, tp, fp = count_correct_words(pred_sentence, source_sentence, n_grams)
            total_true += true 
            total_errors += error
            pred_tp += tp
            pred_fp += fp
        else:
            for word in source_sentence.split():
                if '_' in word: pred_tp += 1
                total_true += 1
    return {
        'Accuracy': total_true / total_words, 
    }

print('Ket qua tach tu khi su dung Maximum Matching la', Token_Eva('tokenizer/MM_tokens.txt'))
print('Ket qua tach tu khi su dung VNCoreNLP la', Token_Eva('tokenizer/vncore_tokens.txt'))