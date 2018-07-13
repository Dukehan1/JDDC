# -*- coding: UTF-8 -*-
import jieba
import numpy as np
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.n_words = 4  # Count PAD, UNK, SOS and EOS
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
_DIGIT_RE = re.compile("\d+")


def word_tokenizer(sentence):
    # word level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    tokens = list(jieba.cut(sentence))
    tokens = [word.encode('utf-8') for word in tokens if word]
    return tokens


def char_tokenizer(sentence):
    # char level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    tokens = list(sentence.decode('utf-8'))
    tokens = [word.encode('utf-8') for word in tokens if word]
    return tokens


def prepare_data(tokenizer):
    # 生成词表，前半部分仅供decoder使用，整体供encoder使用
    lang = Lang('zh-cn')
    files = ['data/train.dec', 'data/train.enc']
    for f in files:
        lines = open(f).read().strip().split('\n')
        for line in lines:
            tokens = tokenizer(line)
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
            tokens = tokenizer(line)
            data[f].append(tokens)

    return lang, data


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return indexes


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(lang, pair[0])
    target_tensor = tensorFromSentence(lang, pair[1])
    return (input_tensor, target_tensor)


lang, data = prepare_data(char_tokenizer)
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


class EncoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input):
        '''
        :param input: (b, l, embed)
        :return:
        '''
        output, h_n = self.gru(input)  # (b, l, hidden * 2), (b, 2, hidden)
        h_n = h_n.view(-1, 1, self.hidden_size * 2)  # (b, 1, hidden * 2)

        return output, h_n

    def pack_unpack(self, input, input_lens):
        packed_input = pack_padded_sequence(input, input_lens, batch_first=True)
        packed_output, h_n = self.forward(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, h_n


class CopynetDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, dec_vocab_size, vocab_size, dropout_p=0.1):
        super(CopynetDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.vocab_size = vocab_size
        self.dec_vocab_size = dec_vocab_size

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.gen_out = nn.Linear(hidden_size, dec_vocab_size)

    def forward(self, input, encoder_outputs, encoder_input_ids, hidden, attention):
        '''
        :param input: (b, 1, embed)
        :param encoder_outputs: (b, l, hidden)
        :param encoder_input_ids: (b, l)
        :param hidden: (b, 1, hidden)
        :param attention: (b, 1, hidden)
        :return:
        '''
        input = self.dropout(input)  # (b, 1, embed)

        input = torch.cat((input, attention), 2)  # (b, 1, hidden + embed)
        _, cur_hidden = self.gru(input, hidden.transpose(0, 1))
        cur_hidden = cur_hidden.transpose(0, 1)  # (b, 1, hidden)

        attn_weights = F.softmax(torch.bmm(self.attn(cur_hidden),
                                           encoder_outputs.transpose(1, 2)), dim=2)  # (b, 1, l)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # (b, 1, hidden)
        cur_attention = torch.tanh(
            self.attn_combine(torch.cat((attn_applied, cur_hidden), 2)))  # (b, 1, hidden_size)

        generate_score = self.gen_out(cur_attention).squeeze(1)  # (b, dec_vocab_size)

        output = F.log_softmax(generate_score, dim=1)  # (b, dec_vocab_size)
        return output, cur_hidden, cur_attention


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


def train(input, input_lens, target, target_lens, embedding, encoder, decoder, optimizer, criterion):
    # Training mode (enable dropout)
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    loss = 0

    b = len(input)

    encoder_input = embedding(input)  # (b, l, embed)
    encoder_outputs, decoder_hidden = encoder.pack_unpack(encoder_input, input_lens)  # (b, l, hidden), (b, 1, hidden)

    decoder_input = torch.zeros(b, 1, dtype=torch.long, device=device).fill_(SOS_token)  # (b, 1)
    decoder_input = embedding(decoder_input)  # (b, 1, embed)

    decoder_attention = torch.zeros(b, 1, decoder.hidden_size, device=device)  # (b, 1, hidden)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(len(target[0])):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, encoder_outputs, input,
                                                                        decoder_hidden, decoder_attention)
            loss += criterion(decoder_output, target[:, di])

            decoder_input = target[:, di].view(-1, 1)  # Teacher forcing
            decoder_input = embedding(decoder_input)  # (b, 1, embed)

    loss.backward()
    # print map(lambda x: x.grad, embedding.parameters())
    # print map(lambda x: x.grad, encoder.parameters())
    # print map(lambda x: x.grad, decoder.parameters())
    clip_grad_norm_(embedding.parameters(), 5)
    clip_grad_norm_(encoder.parameters(), 5)
    clip_grad_norm_(decoder.parameters(), 5)

    optimizer.step()

    return loss.item() / len(target_lens)


def trainIters(embedding, encoder, decoder, n_iters, learning_rate, batch_size, print_every=1000, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    data = []
    optimizer = optim.Adam([{"params": embedding.parameters()}, {"params": encoder.parameters()},
                            {"params": decoder.parameters()}], lr=learning_rate)
    for pair in pairs:
        data.append(tensorsFromPair(pair))
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        print "======================iter%s============================" % iter
        for i, training_pairs in enumerate(get_minibatches(data, batch_size)):
            print "batch: ", i
            # 排序并padding
            training_pairs = sorted(training_pairs, cmp=lambda x, y: cmp(len(x[0]), len(y[0])), reverse=True)
            enc_lens = map(lambda x: len(x[0]), training_pairs)
            dec_lens = map(lambda x: len(x[1]), training_pairs)
            enc_max_len = max(enc_lens)
            dec_max_len = max(dec_lens)
            enc = []
            dec = []
            for t in training_pairs:
                enc.append(t[0] + (enc_max_len - len(t[0])) * [0])
                dec.append(t[1] + (dec_max_len - len(t[1])) * [0])
            enc = torch.tensor(enc, dtype=torch.long, device=device)
            dec = torch.tensor(dec, dtype=torch.long, device=device)

            loss = train(enc, enc_lens, dec, dec_lens, embedding, encoder, decoder, optimizer, criterion)

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


def evaluate(embedding, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        # Inference mode (disable dropout)
        encoder.eval()
        decoder.eval()
        input = torch.tensor([tensorFromSentence(lang, sentence)], dtype=torch.long, device=device)

        encoder_input = embedding(input)  # (1, l, embed)
        encoder_outputs, decoder_hidden = encoder(encoder_input)  # (1, l, hidden), (1, 1, hidden)

        decoder_input = torch.zeros(1, 1, dtype=torch.long, device=device).fill_(SOS_token)  # (1, 1)
        decoder_input = embedding(decoder_input)  # (1, 1, embed)

        decoder_attention = torch.zeros(1, 1, decoder.hidden_size, device=device)  # (1, 1, hidden)

        decoded_words = []

        # TODO: Beam search
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, encoder_outputs, input,
                                                                        decoder_hidden, decoder_attention)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach().view(1, 1)
            decoder_input = embedding(decoder_input)

        return decoded_words


def evaluateRandomly(embedding, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(embedding, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


n_epochs = 100
batch_size = 2
lr = 0.01
embed_size = 200
hidden_size = 256

# 共用一套embedding
embedding = nn.Embedding(lang.n_words, embed_size, padding_idx=0).to(device)
encoder = EncoderRNN(embed_size, lang.n_words, hidden_size / 2).to(device)
attn_decoder = CopynetDecoderRNN(embed_size, hidden_size, lang.n_words_for_decoder, lang.n_words, dropout_p=0.1).to(
    device)

trainIters(embedding, encoder, attn_decoder, n_epochs, lr, batch_size, print_every=1)

evaluateRandomly(embedding, encoder, attn_decoder)
