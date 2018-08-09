# -*- coding: utf-8 -*-
import re
import jieba

# Regular expressions used to tokenize.
_DIGIT_RE = re.compile("\d+")
special_words = [
    '<s>', '#E-s', '[USERID_0]', '[ORDERID_0]', '[数字x]', '[金额x]', '[日期x]', '[时间x]',
    '[电话x]', '[地址x]', '[站点x]', '[姓名x]', '[邮箱x]', '[身份证号x]', '[链接x]', '[组织机构x]',
    '<ORDER_1_N>', '<ORDER_1_1>', '<ORDER_1_2>', '<ORDER_1_3>', '<ORDER_1_4>',
    '<USER_1_N>', '<USER_1_e>', '<USER_1_q>', '<USER_1_r>', '<USER_1_t>', '<USER_1_w>',
    '<USER_2_N>', '<USER_2_0>', '<USER_2_2>', '<USER_2_3>', '<USER_2_4>', '<USER_2_5>', '<USER_2_6>',
]

def prepare_data(tokenizer, files):
    # data_to_token_ids
    data = {}
    for f in files:
        data[f] = []
        lines = files[f]
        for line in lines:
            tokens = tokenizer(line)
            data[f].append(tokens)

    return data

def word_tokenizer(sentence):
    # word level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    chunks = re.split(r'(' + '|'.join(special_words).replace('[', '\[').replace(']', '\]') + ')', sentence)
    tokens = []
    for c in chunks:
        if c not in special_words:
            tokens.extend(list(jieba.cut(c)))
        else:
            tokens.append(c)
    tokens = [word for word in tokens if word]
    return tokens

def char_tokenizer(sentence):
    # char level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    tokens = list(sentence)
    tokens = [word for word in tokens if word]
    return tokens

ENC_MAX_LEN = 1000
DEC_MAX_LEN = 100

def process():
    # 从10份中取一份作dev
    dev_id = 0
    files = {
        'trainAnswers': [],
        'devAnswers': [],
        'trainQuestions': [],
        'devQuestions': []
    }
    train_idx = list(filter(lambda x: x != dev_id, range(10)))
    for i in train_idx:
        files['trainAnswers'].extend(
            open('data/answers-' + str(i) + '.txt', encoding="utf-8").read().strip().split('\n'))
        files['trainQuestions'].extend(
            open('data/questions-' + str(i) + '.txt', encoding="utf-8").read().strip().split('\n'))
    files['devAnswers'].extend(
        open('data/answers-' + str(dev_id) + '.txt', encoding="utf-8").read().strip().split('\n'))
    files['devQuestions'].extend(
        open('data/questions-' + str(dev_id) + '.txt', encoding="utf-8").read().strip().split('\n'))

    data = prepare_data(word_tokenizer, files)

    trainAnswers = open("opennmt-kb/train.txt.tgt", 'w', encoding="utf-8")
    devAnswers = open("opennmt-kb/val.txt.tgt", 'w', encoding="utf-8")
    trainQuestions = open("opennmt-kb/train.txt.src", 'w', encoding="utf-8")
    devQuestions = open("opennmt-kb/val.txt.src", 'w', encoding="utf-8")
    for i in range(len(data['trainQuestions'])):
        if len(data['trainQuestions'][i]) > ENC_MAX_LEN:
            data['trainQuestions'][i] = data['trainQuestions'][i][-ENC_MAX_LEN:]
            # print(data['trainQuestions'][i])
            # print(len(data['trainQuestions'][i]))
        if len(data['trainAnswers'][i]) > DEC_MAX_LEN:
            data['trainAnswers'][i] = data['trainAnswers'][i][:DEC_MAX_LEN]
            # print(data['trainAnswers'][i])
            # print(len(data['trainAnswers'][i]))
        trainQuestions.write(' '.join(data['trainQuestions'][i]) + '\n')
        trainAnswers.write(' '.join(data['trainAnswers'][i]) + '\n')
    for i in range(len(data['devQuestions'])):
        if len(data['devQuestions'][i]) > ENC_MAX_LEN:
            data['devQuestions'][i] = data['devQuestions'][i][-ENC_MAX_LEN:]
            # print(data['devQuestions'][i])
            # print(len(data['devQuestions'][i]))
        if len(data['devAnswers'][i]) > DEC_MAX_LEN:
            data['devAnswers'][i] = data['devAnswers'][i][:DEC_MAX_LEN]
            # print(data['devAnswers'][i])
            # print(len(data['devAnswers'][i]))
        devQuestions.write(' '.join(data['devQuestions'][i]) + '\n')
        devAnswers.write(' '.join(data['devAnswers'][i]) + '\n')


if __name__ == "__main__":
    process()