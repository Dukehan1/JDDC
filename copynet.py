# -*- coding: UTF-8 -*-
from io import open
import jieba
import numpy as np
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


UNK_token = 0
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "UNK", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count UNK, SOS and EOS
        self.n_words_for_decoder = self.n_words

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def updateDecoderWords(self):
        # 记录Decoder词表的大小
        self.n_words_for_decoder = self.n_words


# Regular expressions used to tokenize.
_DIGIT_RE = re.compile("\d")


def basic_tokenizer(sentence):
    # char level
    words = []
    sentence = list(jieba.cut(sentence))
    for i in range(len(sentence)):
        words.extend(sentence[i])
    return [w.encode('utf-8') for w in words if w]


def prepare_data(tokenizer=None):
    # 生成词表，前半部分仅供decoder使用，整体供encoder使用
    lang = Lang('zh-cn')
    files = ['data/train.dec', 'data/train.enc']
    for f in files:
        lines = open(f).read().strip().split('\n')
        for line in lines:
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            tokens = map(lambda w: re.sub(_DIGIT_RE, "0", w), tokens)
            lang.addSentence(tokens)
        # 记录Decoder词表的大小
        if f == 'data/train.dec':
            lang.updateDecoderWords()

    # data_to_token_ids
    files = ['data/train.enc', 'data/train.dec', 'data/test.enc', 'data/test.dec']
    data = {}
    for f in files:
        data[f] = []
        lines = open(f).read().strip().split('\n')
        for line in lines:
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            tokens = list(map(lambda w: re.sub(_DIGIT_RE, "0", w), tokens))
            data[f].append(tokens)

    return lang, data


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else 0 for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(lang, pair[0])
    target_tensor = tensorFromSentence(lang, pair[1])
    return (input_tensor, target_tensor)


lang, data = prepare_data()
pairs = []
length = []
for i in range(5):
    # for i in range(len(data['data/train.enc'])):
    pairs.append((data['data/train.enc'][i], data['data/train.dec'][i]))
    length.append(len(data['data/train.enc'][i]))
print 'max input length: ', max(length)
# print pairs[0]
# print tensorsFromPair(pairs[0])


# 实际输入的序列会多一个终止token
MAX_LENGTH = max(length) + 1


class CopyNetWrapperState():
    def __init__(self, last_id, hidden, attention):
        self.last_id = last_id
        self.hidden = hidden
        self.attention = attention


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        '''
        :param input: (1, )
        :param hidden: (1, 1, hidden_size)
        :return:
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class CopynetDecoderRNN(nn.Module):
    def __init__(self, hidden_size, dec_vocab_size,
                 vocab_size, embedding, dropout_p=0.1, max_length=MAX_LENGTH):
        super(CopynetDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.dec_vocab_size = dec_vocab_size

        self.embedding = embedding
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size * 3, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gen_out = nn.Linear(self.hidden_size, self.dec_vocab_size)
        self.copy_out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, encoder_input_ids,
                copy_tuple, encoder_outputs):
        '''
        :param input: (1, )
        :param encoder_outputs: (max_length, hidden_size)
        :param encoder_input_ids: (length, 1)
        :return:
        '''
        last_id = copy_tuple.last_id  # scalar
        hidden = copy_tuple.hidden  # (1, 1, hidden_size)
        attention = copy_tuple.attention  # (1, hidden_size)

        input_length = encoder_input_ids.size(0)
        mask = torch.eq(torch.empty_like(encoder_input_ids).fill_(last_id), encoder_input_ids).float()
        mask_sum = torch.sum(mask, dim=0)
        if mask_sum.item() != 0:
            mask = mask / mask_sum.item()
        rou = torch.cat((mask, torch.zeros(self.max_length - input_length, 1)), 0)  # (max_length, 1)
        selective_read = torch.matmul(rou.transpose(0, 1), encoder_outputs)  # (1, hidden_size)

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        input = torch.cat((embedded[0], selective_read, attention), 1).view(1, 1, -1)
        _, cur_hidden = self.gru(input, hidden)  # (1, hidden_size)

        attn_weights = F.softmax(torch.bmm(self.attn(cur_hidden[0]).view(1, 1, -1),
                                           encoder_outputs.unsqueeze(0).transpose(1, 2)), dim=2)  # (1, 1, max_length)
        attn_applied = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))  # (1, 1, hidden_size)
        cur_attention = torch.tanh(self.attn_combine(torch.cat((attn_applied[0], cur_hidden[0]), 1)))  # (1, hidden_size)

        generate_score = torch.exp(self.gen_out(cur_attention).view(-1, 1))  # (dec_vocab_size, 1)
        '''
        copy_weights = torch.sigmoid(self.copy_out(encoder_outputs))  # (max_length, hidden_size)
        copy_score = torch.exp(torch.matmul(copy_weights, cur_attention.transpose(0, 1)).view(-1, 1))  # (max_length, 1)

        encoder_input_mask = torch.zeros(input_length, self.vocab_size).scatter_(1, encoder_input_ids, 1)  # (length, vocab_size)
        padding_encoder_input_mask = torch.cat((encoder_input_mask,
                                                torch.zeros(self.max_length - input_length, self.vocab_size)), 0)  # (max_length, vocab_size)
        prob_c_one_hot = copy_score * padding_encoder_input_mask  # (max_length, vocab_size)

        gen_output_mask = torch.zeros(self.dec_vocab_size, self.vocab_size). \
            scatter_(1, torch.tensor(range(self.dec_vocab_size), dtype=torch.long).view(-1, 1), 1)  # (dec_vocab_size, vocab_size)
        prob_g_one_hot = generate_score * gen_output_mask  # (dec_vocab_size, vocab_size)

        prob_total = torch.cat((prob_g_one_hot, prob_c_one_hot), 0)  # (dec_vocab_size + max_length, vocab_size)
        output = torch.sum(prob_total, dim=0)  # (vocab_size, )
        output = torch.log(output / torch.sum(output, dim=0))  # (vocab_size, )
        '''
        output = F.log_softmax(generate_score.view(-1))  # (dec_vocab_size, )

        _, cur_last_id = torch.max(output, 0)
        cur_last_id = cur_last_id.detach()  # detach from history

        cur_copy = CopyNetWrapperState(cur_last_id, cur_hidden, cur_attention)
        return output.view(1, -1), cur_copy

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


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


teacher_forcing_ratio = 1.0


def train(training_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Training mode (enable dropout)
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    for training_pair in training_pairs:
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        encoder_hidden = encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        init_attention = torch.zeros(1, encoder.hidden_size, device=device)
        copy_tuple = CopyNetWrapperState(-1, encoder_hidden, init_attention)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, copy_tuple = decoder(decoder_input, input_tensor, copy_tuple, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])  # 去掉decoder_output中的-inf

                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, copy_tuple = decoder(decoder_input, input_tensor, copy_tuple, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])  # 去掉decoder_output中的-inf

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                if decoder_input.item() == EOS_token:
                    break

    loss.backward()
    # print map(lambda x: x.grad, encoder.parameters())
    # print map(lambda x: x.grad, decoder.parameters())
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def trainIters(encoder, decoder, n_iters, learning_rate, batch_size, print_every=1000, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    data = []
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    for pair in pairs:
        data.append(tensorsFromPair(pair))
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        print "======================iter%s============================" % iter
        for i, training_pairs in enumerate(get_minibatches(data, batch_size)):
            print "batch: ", i
            loss = train(training_pairs, encoder, decoder,
                         encoder_optimizer, decoder_optimizer, criterion)

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


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        # Inference mode (disable dropout)
        encoder.eval()
        decoder.eval()
        input_tensor = tensorFromSentence(lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoded_words = []
        init_attention = torch.zeros(1, encoder.hidden_size)
        copy_tuple = CopyNetWrapperState(-1, encoder_hidden, init_attention)

        # TODO: Beam search
        for di in range(max_length):
            decoder_output, copy_tuple = decoder(
                decoder_input, input_tensor, copy_tuple, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


n_epochs = 100
batch_size = 2
lr = 0.001
hidden_size = 256


# 共用一套embedding
embedding = nn.Embedding(lang.n_words, hidden_size).to(device)
encoder1 = EncoderRNN(lang.n_words, hidden_size, embedding).to(device)
attn_decoder1 = CopynetDecoderRNN(hidden_size, lang.n_words_for_decoder, lang.n_words, embedding, dropout_p=0.1).to(
    device)


trainIters(encoder1, attn_decoder1, n_epochs, lr, batch_size, print_every=1)


evaluateRandomly(encoder1, attn_decoder1)
