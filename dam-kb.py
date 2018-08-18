# -*- coding: UTF-8 -*-
import codecs

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
from torch.nn.utils import clip_grad_norm_
import os
import pickle
import math
# from gensim import corpora, models, similarities

from dataToOpenNMT import sentence_with_kb, word_tokenizer

device_cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

VECTOR_DIR = 'embed/sgns.merge.word'


def load_stopwords():
    # load stopwords from file
    stopwords = []
    file_obj = codecs.open('opennmt-kb/stopword.txt', 'r', 'utf-8')
    while True:
        line = file_obj.readline()
        line = line.strip('\r\n')
        if not line:
            break
        stopwords.append(line)
    stopwords = set(stopwords)
    file_obj.close()
    return stopwords


def init_embedding(embed_size, n_word, word2index):
    if os.path.exists('model_dam/pretrained'):
        t = open('model_dam/pretrained', 'rb')
        embeddings = pickle.load(t)
        t.close()
    else:
        print("Loading pretrained embeddings...")
        start = time.time()

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype=np.float32)

        embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(VECTOR_DIR, encoding='utf-8') if
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

        t = open('model_dam/pretrained', 'wb')
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
    result = [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence]
    return result


class AttentiveModule(nn.Module):
    def __init__(self, embed_size, max_length):
        super(AttentiveModule, self).__init__()
        self.embed_size = embed_size

        self.norm1 = nn.LayerNorm((max_length, embed_size))
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm((max_length, embed_size))

    def forward(self, query, key, value):
        '''
        :param query: (b, l, embed)
        :param key: (b, l, embed)
        :param value: (b, l, embed)
        :return:
        '''
        att_q_k = torch.softmax(torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.embed_size), dim=2)  # (b, l, l)
        v_att = torch.bmm(att_q_k, value)  # (b, l, embed)
        input = v_att + query  # (b, l, embed)
        input = self.norm1(input)
        output = self.ffn(input)
        result = input + output
        result = self.norm2(result)
        return result


class DAM(nn.Module):
    def __init__(self, embed_size, max_num_utterance, max_length, max_stacks):
        super(DAM, self).__init__()
        self.embed_size = embed_size
        self.max_num_utterance = max_num_utterance
        self.max_length = max_length
        self.max_stacks = max_stacks

        self.am = AttentiveModule(self.embed_size, self.max_length)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=max_num_utterance, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3)),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(3, 3, 3)),
        )
        self.final = nn.Linear(256, 2)

    def forward(self, utterances, response):
        '''
        :param utterances: (b, max_num_utterance, l, embed)
        :param response: (b, l, embed)
        :return:
        '''
        b = response.size(0)
        utterances = utterances.transpose(0, 1)

        R = [response]
        for _ in range(self.max_stacks):
            response = self.am(response, response, response)
            R.append(response)

        cube = []

        for utterance in utterances:
            U = [utterance]
            for _ in range(self.max_stacks):
                utterance = self.am(utterance, utterance, utterance)
                U.append(utterance)

            self_attention = []
            cross_attention = []
            for i in range(len(U)):
                self_attention.append(torch.bmm(U[i], R[i].transpose(1, 2)))
                cross_attention.append(torch.bmm(self.am(U[i], R[i], R[i]), self.am(R[i], U[i], U[i]).transpose(1, 2)))
            cube.append(torch.stack(self_attention + cross_attention, 1))

        cube = torch.stack(cube, 1)  # (b, max_num_utterance, 2 * (max_stacks + 1), max_length, max_length)
        v = self.conv(cube)
        output = F.log_softmax(self.final(v.view(b, -1)), dim=1)  # (b, 2)
        return output


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


def train(utterances, response, label, embedding, dam, optimizer, criterion):
    # Training mode (enable dropout)
    dam.train()
    optimizer.zero_grad()

    utterances = embedding(utterances)
    response = embedding(response)
    output = dam(utterances, response)
    loss = criterion(output, label)

    loss.backward()
    # print(list(map(lambda x: x.grad, embedding.parameters())))
    # print(list(map(lambda x: x.grad, dam.parameters())))
    clip_grad_norm_(embedding.parameters(), 2)
    clip_grad_norm_(dam.parameters(), 2)

    optimizer.step()

    return loss.data.item()


def inference(utterances, response, embedding, dam):
    with torch.no_grad():
        # Inference mode (disable dropout)
        dam.eval()

        utterances = embedding(utterances)
        response = embedding(response)
        output = dam(utterances, response)
        output = output[:, 1]
        output = output.data.cpu().numpy()

        return output

'''
def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)
'''


def trainIters(vocab, embedding, dam, optimizer, train_examples, dev_examples, n_iters, learning_rate, batch_size,
               infer_batch_size, print_every=1000, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_score = 0.0
    criterion = nn.NLLLoss()

    # 在有Model时预先生成best_score
    print("============evaluate_start==================")
    score = evaluate(embedding, dam, dev_examples, infer_batch_size)
    print('R10@1_socre: {0}'.format(score))
    if score >= best_score:
        best_score = score
        torch.save(dam.state_dict(), 'model_dam/dam')
        torch.save(embedding.state_dict(), 'model_dam/embedding')
        torch.save(optimizer.state_dict(), 'model_dam/optimizer')
        print('new model_dam saved.')
    print("==============evaluate_end==================")

    for iter in range(1, n_iters + 1):
        print("======================iter%s============================" % iter)
        for i, (utterances, response, label) in enumerate(get_minibatches(train_examples, batch_size)):

            # print("batch: ", i)
            # 构建负例（1:1）（实际训练数据大小为batch * 2）
            neg_response = []
            for idx in range(len(label)):
                neg = np.random.randint(0, len(train_examples[1]))
                neg_response.append(train_examples[1][neg])

            utterances = torch.tensor(utterances + utterances, dtype=torch.long, device=device_cuda)
            response = torch.tensor(response + neg_response, dtype=torch.long, device=device_cuda)
            label = torch.tensor(label + [0] * len(label), dtype=torch.long, device=device_cuda)

            loss = train(utterances, response, label, embedding, dam, optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        print("============evaluate_start==================")
        score = evaluate(embedding, dam, dev_examples, infer_batch_size)
        print('R10@1_socre: {0}'.format(score))
        if score >= best_score:
            best_score = score
            torch.save(dam.state_dict(), 'model_dam/dam')
            torch.save(embedding.state_dict(), 'model_dam/embedding')
            torch.save(optimizer.state_dict(), 'model_dam/optimizer')
            print('new model_dam saved.')
        print("==============evaluate_end==================")


def evaluate(embedding, dam, dev_examples, infer_batch_size):
    predict = []
    for i in range(0, len(dev_examples[0]), infer_batch_size):
        # print("batch: ", int(i / infer_batch_size))
        utterances = dev_examples[0][i:i + infer_batch_size]
        response = dev_examples[1][i:i + infer_batch_size]

        utterances = torch.tensor(utterances, dtype=torch.long, device=device_cuda)
        response = torch.tensor(response, dtype=torch.long, device=device_cuda)

        predict.extend(inference(utterances, response, embedding, dam))

    score = ComputeR10_1(predict, dev_examples[2])
    return score


max_length = 50
max_num_utterance = 6
max_stacks = 5

n_epochs = 20
lr = 0.001
batch_size = 64
infer_batch_size = 1000
embed_size = 300
cut = 8

K = 15


def gen_data(vocab_word, q, a, n):
    '''
    截断数据
    '''
    utterances_list = []
    response_list = []
    for i in range(len(q)):
        utterances = []
        temp = []
        j = 0
        chunk = q[i]
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
        temp = utterances[1:]
        temp.reverse()
        utterances = utterances[:1] + temp
        utterances = utterances + (max_num_utterance - len(utterances)) * [[]]
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


def run_train():
    data = {
        'trainAnswers': [],
        'devAnswers': [],
        'trainQuestions': [],
        'devQuestions': []
    }
    data['trainAnswers'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb/train.txt.tgt', encoding="utf-8").read().strip().split('\n')))
    data['trainQuestions'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb/train.txt.src', encoding="utf-8").read().strip().split('\n')))
    data['devAnswers'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb/val.txt.tgt', encoding="utf-8").read().strip().split('\n')))
    data['devQuestions'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb/val.txt.src', encoding="utf-8").read().strip().split('\n')))

    # for debug
    data['trainAnswers'] = data['trainAnswers'][:5]
    data['trainQuestions'] = data['trainQuestions'][:5]
    data['devAnswers'] = [list(x) for x in data['trainAnswers']]
    data['devQuestions'] = [list(x) for x in data['trainQuestions']]

    # 生成词表（word）
    if os.path.exists('model_dam/vocab_word'):
        t = open('model_dam/vocab_word', 'rb')
        vocab_word = pickle.load(t)
        t.close()
    else:
        vocab_word = prepare_vocabulary(data, cut=cut)
        t = open('model_dam/vocab_word', 'wb')
        pickle.dump(vocab_word, t)
        t.close()
    print("========================word===========================")
    print('dec_vocab_size: ', vocab_word.n_words_for_decoder)
    print('vocab_size: ', vocab_word.n_words)
    print('max_word_length: ', max(map(lambda x: len(x), vocab_word.word2index)))

    # 生成数据（截断，生成负例）
    if os.path.exists('model_dam/data'):
        t = open('model_dam/data', 'rb')
        train_examples, dev_examples = pickle.load(t)
        t.close()
    else:
        train_examples = gen_data(vocab_word, data['trainQuestions'], data['trainAnswers'], 1)
        dev_examples = gen_data(vocab_word, data['devQuestions'], data['devAnswers'], 10)
        t = open('model_dam/data', 'wb')
        pickle.dump((train_examples, dev_examples), t)
        t.close()
    print("========================dataset===========================")
    print('train: ', len(train_examples[0]), len(train_examples[1]), len(train_examples[2]))
    print('dev: ', len(dev_examples[0]), len(dev_examples[1]), len(dev_examples[2]))

    embed = init_embedding(embed_size, vocab_word.n_words, vocab_word.word2index)
    embedding = nn.Embedding(vocab_word.n_words, embed_size, padding_idx=0).from_pretrained(embed, freeze=False)
    dam = DAM(embed_size, max_num_utterance, max_length, max_stacks)

    embedding = torch.nn.DataParallel(embedding).to(device_cuda)
    dam = torch.nn.DataParallel(dam).to(device_cuda)

    if os.path.isfile('model_dam/embedding'):
        embedding.load_state_dict(torch.load('model_dam/embedding'))

    if os.path.isfile('model_dam/dam'):
        dam.load_state_dict(torch.load('model_dam/dam'))

    optimizer = optim.Adam([{"params": embedding.parameters()}, {"params": dam.parameters()}], lr=lr, amsgrad=True)

    if os.path.isfile('model_dam/optimizer'):
        optimizer.load_state_dict(torch.load('model_dam/optimizer'))

    trainIters(vocab_word, embedding, dam, optimizer, train_examples, dev_examples, n_epochs, lr, batch_size,
               infer_batch_size, print_every=1)


if __name__ == "__main__":
    run_train()
