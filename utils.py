# -*- coding: UTF-8 -*-
import os
import pickle
import time
import torch
import numpy as np
import re
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device_cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

VECTOR_WORD_DIR = 'embed/sgns.merge.word'
VECTOR_CHAR_DIR = 'embed/sgns.merge.char'


def init_embedding(embed_size, n_word, word2index, is_char=False):
    if is_char:
        embed_path = 'embed/pretrained_char'
    else:
        embed_path = 'embed/pretrained_word'
    if os.path.exists(embed_path):
        t = open(embed_path, 'rb')
        embeddings = pickle.load(t)
        t.close()
    else:
        print("Loading pretrained embeddings...")
        start = time.time()

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype=np.float32)

        if is_char:
            embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(VECTOR_CHAR_DIR, encoding='utf-8') if
                                   len(o.rstrip().rsplit(' ')) != 2)
        else:
            embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(VECTOR_WORD_DIR, encoding='utf-8') if
                                   len(o.rstrip().rsplit(' ')) != 2)
        print(len(embeddings_dict))

        # print 'no pretrained: '
        embeddings = np.random.randn(n_word, embed_size).astype(np.float32)
        for word, i in word2index.items():
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embeddings[i] = embedding_vector
                print('in: ', word)
            else:
                print('out: ', word)
        print("took {:.2f} seconds\n".format(time.time() - start))

        t = open(embed_path, 'wb')
        pickle.dump(embeddings, t)
        t.close()
    return torch.tensor(embeddings, device=device_cpu)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.word2count = {}
        self.index2word = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        self.n_words = 4  # Count PAD, UNK, SOS and EOS
        self.n_words_for_decoder = self.n_words

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word.append(word)
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def updateDecoderWords(self):
        # 记录Decoder词表的大小
        self.n_words_for_decoder = self.n_words


def prepare_vocabulary(data, cut=3):
    # 生成词表，前半部分仅供decoder使用，整体供encoder使用
    lang = Lang('zh-cn')
    # 仅使用训练集统计
    data = {
        'trainAnswers': data['trainAnswers'],
        'trainQuestions': data['trainQuestions']
    }
    for f in data:
        lines = data[f]
        for line in lines:
            lang.addSentence(line)
        # 记录Decoder词表的大小
        if f == 'trainAnswers':
            lang.updateDecoderWords()

    # 削减词汇表
    dec = []
    rest = []
    for k, v in enumerate(lang.index2word):
        if v in lang.word2count:
            if k < lang.n_words_for_decoder:
                dec.append((v, lang.word2count[v]))
            else:
                rest.append((v, lang.word2count[v]))
    dec = list(filter(lambda x: x[1] > cut, dec))
    rest = list(filter(lambda x: x[1] > cut, rest))
    lang.n_words_for_decoder = 4 + len(dec)
    lang.n_words = 4 + len(dec) + len(rest)
    lang.word2index = {}
    lang.word2count = {}
    lang.index2word = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    for i in dec + rest:
        lang.word2index[i[0]] = len(lang.index2word)
        lang.word2count[i[0]] = i[1]
        lang.index2word.append(i[0])

    return lang


def indexesFromSentence(lang, sentence):
    '''
    检索
    '''
    result = [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence]
    return result


def indexesFromSentenceInSeq(lang, sentence):
    '''
    生成
    '''
    result = [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence]
    result.append(EOS_token)
    return result


def indexesFromPair(lang, pair):
    input = indexesFromSentenceInSeq(lang, pair[0])
    target = indexesFromSentenceInSeq(lang, pair[1])
    return input, target


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:
        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...
    Or with multiple data sources:
        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...
    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.
    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def ComputeR10_1(scores, labels, count=10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total + 1
            sublist = scores[i+1:i+count]
            flag = True
            for j in sublist:
                if scores[i] <= j:
                    flag = False
                    break
            if flag:
                correct = correct + 1
    return float(correct) / total


def gen_data(vocab_word, q, a, n, max_length, max_num_utterance):
    '''
    截断数据
    '''
    utterances_list = []
    response_list = []
    for i in range(len(q)):
        utterances = []
        temp = []
        j = 0
        chunk = list(q[i])  # 不要让数组对象重复引用
        chunk.reverse()
        for w in chunk:
            if w == '<s>':
                j += 1
            if j < 6:
                temp.insert(0, w)
            else:
                if w == '<s>':
                    utterances.append(temp)
                    temp = []
                else:
                    temp.insert(0, w)
        utterances.append(temp)
        utterances = utterances[:max_num_utterance]
        utterances = utterances + (max_num_utterance - len(utterances)) * [[]]
        utterances.reverse()
        utterances = list(map(lambda x: indexesFromSentence(vocab_word, x)[:max_length], utterances))
        utterances = list(map(lambda x: x + (max_length - len(x)) * [0], utterances))
        utterances_list.append(utterances)
        response = a[i][:max_length]
        response = indexesFromSentence(vocab_word, response)
        response = response + (max_length - len(response)) * [0]
        response_list.append(response)
    label_list = [1] * len(q)

    # 构建负例（1:n-1），n=1时不构建
    if n == 1:
        return [utterances_list, response_list, label_list]

    new_utterances_list = []
    new_response_list = []
    for i in range(len(q)):
        neg_index = []
        while True:
            temp = np.random.randint(0, len(q))
            if temp != i:
                neg_index.append(temp)
            if len(neg_index) >= n - 1:
                break
        new_utterances_list.extend([utterances_list[i]] * n)
        new_response_list.append(response_list[i])
        for j in neg_index:
            new_response_list.append(response_list[j])
    new_label_list = ([1] + [0] * (n - 1)) * len(q)
    return [new_utterances_list, new_response_list, new_label_list]


def bleu(gold, predict):
    chencherry = SmoothingFunction()
    score = []
    for i in range(len(gold)):
        bleuScore = sentence_bleu([list(gold[i])], list(predict[i]), weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=chencherry.method1)
        score.append(bleuScore)
    # 最终得分
    scoreFinal = sum(score) / float(len(score))
    # 最终得分精确到小数点后6位
    precisionScore = round(scoreFinal, 6)
    return precisionScore


def cut_utterances(chunk, max_num_utterance, max_seq_length):
    '''
    裁剪并填充utterances（在生成模型中使用）
    '''
    chunk = list(chunk)  # 不要让数组对象重复引用
    utterances = []
    temp = []
    j = 0
    chunk.reverse()
    for w in chunk:
        if w == '<s>':
            j += 1
        if j < 6:
            temp.insert(0, w)
        else:
            if w == '<s>':
                utterances.append(temp)
                temp = []
            else:
                temp.insert(0, w)
    utterances.append(temp)
    utterances = utterances[:max_num_utterance]
    utterances = utterances + (max_num_utterance - len(utterances)) * [[]]
    utterances.reverse()
    utterances = list(map(lambda x: x[:max_seq_length], utterances))
    utterances = list(map(lambda x: x + (max_seq_length - len(x)) * ['<PAD>'], utterances))
    result = []
    for u in utterances:
        result.extend(u)
    return result


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


def char_tokenizer(sentence):
    # char level
    sentence = re.sub(_ORDER_RE, "[ORDERID_0]", sentence)
    sentence = re.sub(_USER_RE, "[USERID_0]", sentence)
    sentence = re.sub(_URL_RE, "https://item.jd.com/[数字x].html", sentence)
    chunks = re.split(r'(' + '|'.join(special_words).replace('[', '\[').replace(']', '\]') + ')', sentence)
    tokens = []
    for c in chunks:
        if c not in special_words:
            tokens.extend(list(c))
        else:
            tokens.append(c)
    # 统一表示，将' '全部替换成''
    tokens = [word.strip() for word in tokens if word]
    return tokens


def process(dev_id):
    # 从10份中取一份作dev
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

    trainAnswers = open("opennmt-kb-" + str(dev_id) + "/train.txt.tgt", 'w', encoding="utf-8")
    devAnswers = open("opennmt-kb-" + str(dev_id) + "/val.txt.tgt", 'w', encoding="utf-8")
    trainQuestions = open("opennmt-kb-" + str(dev_id) + "/train.txt.src", 'w', encoding="utf-8")
    devQuestions = open("opennmt-kb-" + str(dev_id) + "/val.txt.src", 'w', encoding="utf-8")

    for i in range(len(data['trainQuestions'])):
        trainQuestions.write(' '.join(data['trainQuestions'][i]) + '\n')
        trainAnswers.write(' '.join(data['trainAnswers'][i]) + '\n')
    for i in range(len(data['devQuestions'])):
        devQuestions.write(' '.join(data['devQuestions'][i]) + '\n')
        devAnswers.write(' '.join(data['devAnswers'][i]) + '\n')


def process_char(dev_id):
    # 从10份中取一份作dev
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

    data = prepare_data(char_tokenizer, files)

    trainAnswers = open("opennmt-kb-char-" + str(dev_id) + "/train.txt.tgt", 'w', encoding="utf-8")
    devAnswers = open("opennmt-kb-char-" + str(dev_id) + "/val.txt.tgt", 'w', encoding="utf-8")
    trainQuestions = open("opennmt-kb-char-" + str(dev_id) + "/train.txt.src", 'w', encoding="utf-8")
    devQuestions = open("opennmt-kb-char-" + str(dev_id) + "/val.txt.src", 'w', encoding="utf-8")

    for i in range(len(data['trainQuestions'])):
        trainQuestions.write(' '.join(data['trainQuestions'][i]) + '\n')
        trainAnswers.write(' '.join(data['trainAnswers'][i]) + '\n')
    for i in range(len(data['devQuestions'])):
        devQuestions.write(' '.join(data['devQuestions'][i]) + '\n')
        devAnswers.write(' '.join(data['devAnswers'][i]) + '\n')


if __name__ == "__main__":
    process_char(0)
