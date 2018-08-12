# -*- coding: utf-8 -*-
import re
import jieba

# Regular expressions used to tokenize.
_ORDER_RE = re.compile("\[(ORDERID_\d+)\]")
_USER_RE = re.compile("\[(USERID_\d+)\]")
_URL_RE = re.compile("https?://item.jd.com/(\d+|\[数字x\]).html")
special_words = [
    '<s>', '#E-s', 'https://item.jd.com/[数字x].html',
    '[USERID_0]', '[ORDERID_0]', '[数字x]', '[金额x]', '[日期x]', '[时间x]',
    '[电话x]', '[地址x]', '[站点x]', '[姓名x]', '[邮箱x]', '[身份证号x]', '[链接x]', '[组织机构x]',
    '<ORDER_1_N>', '<ORDER_1_1>', '<ORDER_1_2>', '<ORDER_1_3>', '<ORDER_1_4>',
    '<USER_1_N>', '<USER_1_e>', '<USER_1_q>', '<USER_1_r>', '<USER_1_t>', '<USER_1_w>',
    '<USER_2_N>', '<USER_2_0>', '<USER_2_2>', '<USER_2_3>', '<USER_2_4>', '<USER_2_5>', '<USER_2_6>',
]

# 读取knowledge
order = {}
user = {}
sku = {}
user_file = open("preliminaryData/user.txt", encoding="utf-8").read().strip().split('\n')
sku_file = open("preliminaryData/ware_p.txt", encoding="utf-8").read().strip().split('\n')[1:]
order_file = open("preliminaryData/order.txt", encoding="utf-8").read().strip().split('\n')[1:]
for w in sku_file:
    w = w.split("|||")
    sku[w[0]] = [w[0], w[1], w[2]]
for w in user_file:
    w = w.split("\t")
    user[w[0]] = [w[0], w[1], w[2], w[3]]
for w in order_file:
    w = w.split("\t")
    if w[0] in sku:
        order[w[0]].append([w[0], w[1], w[2], w[3], w[4]])
    else:
        order[w[0]] = [[w[0], w[1], w[2], w[3], w[4]]]


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


def sentence_with_kb(sentence):
    # 抽取KB
    chunks = re.split(r'<s>', sentence)
    chunks = chunks[:-1]

    order_1 = '<ORDER_1_N>'
    user_1 = '<USER_1_N>'
    user_2 = '<USER_2_N>'
    sku_1 = ''
    sku_2 = ''
    for c in chunks:
        if re.search(_USER_RE, c):
            user_id = re.search(_USER_RE, c).group(1)
            if user_id in user:
                user_1 = '<USER_1_' + user[user_id][1] + '>'
                user_2 = '<USER_2_' + user[user_id][3] + '>'
        if re.search(_URL_RE, c):
            sku_id = re.search(_URL_RE, c).group(1)
            if sku_id in sku:
                sku_1 = sku[sku_id][1]
                sku_2 = sku[sku_id][2]
        if re.search(_ORDER_RE, c):
            order_id = re.search(_ORDER_RE, c).group(1)
            if order_id in order:
                sku_1s = []
                sku_2s = []
                for term in order[order_id]:
                    order_1 = '<ORDER_1_' + term[4] + '>'
                    sku_1s.append(sku[term[2]][1])
                    sku_2s.append(sku[term[2]][2])
                sku_1 = '|'.join(sku_1s)
                sku_2 = '|'.join(sku_2s)

    result = '<s>'.join(chunks + [order_1, user_1, user_2, sku_1, sku_2]) + '<s>'
    return result


def word_tokenizer(sentence):
    # word level
    sentence = re.sub(_ORDER_RE, "[ORDERID_0]", sentence)
    sentence = re.sub(_USER_RE, "[USERID_0]", sentence)
    sentence = re.sub(_URL_RE, "https://item.jd.com/[数字x].html", sentence)
    chunks = re.split(r'(' + '|'.join(special_words).replace('[', '\[').replace(']', '\]') + ')', sentence)
    tokens = []
    for c in chunks:
        if c not in special_words:
            tokens.extend(list(jieba.cut(c)))
        else:
            tokens.append(c)
    # 统一表示，将' '全部替换成''
    tokens = [word.strip() for word in tokens if word]
    return tokens


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
        trainQuestions.write(' '.join(data['trainQuestions'][i]) + '\n')
        trainAnswers.write(' '.join(data['trainAnswers'][i]) + '\n')
    for i in range(len(data['devQuestions'])):
        devQuestions.write(' '.join(data['devQuestions'][i]) + '\n')
        devAnswers.write(' '.join(data['devAnswers'][i]) + '\n')


if __name__ == "__main__":
    process()