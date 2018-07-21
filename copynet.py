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
import subprocess

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


# Regular expressions used to tokenize.
_DIGIT_RE = re.compile("\d+")


def word_tokenizer(sentence):
    # word level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    tokens = list(jieba.cut(sentence))
    tokens = [word for word in tokens if word]
    return tokens


def char_tokenizer(sentence):
    # char level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    tokens = list(sentence)
    tokens = [word for word in tokens if word]
    return tokens


def prepare_vocabulary(tokenizer, files, cut=3):
    # 生成词表，前半部分仅供decoder使用，整体供encoder使用
    lang = Lang('zh-cn')
    # 仅使用训练集统计
    files = {
        'trainAnswers': files['trainAnswers'],
        'trainQuestions': files['trainQuestions']
    }
    for f in files:
        lines = files[f]
        for line in lines:
            tokens = tokenizer(line)
            lang.addSentence(tokens)
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


def prepare_data(tokenizer, files):
    # data_to_token_ids
    data = {}
    for f in files:
        data[f] = []
        tokenized_sentence = []
        lines = files[f]
        for line in lines:
            tokens = tokenizer(line)
            data[f].append(tokens)
            tokenized_sentence.append(' '.join(tokens))
        # 记录devAnswers，供bleu使用
        if f == 'devAnswers':
            with open('bleu/gold', 'w', encoding='utf-8') as gw:
                gw.write('\n'.join(tokenized_sentence))

    return data


def indexesFromSentence(lang, sentence):
    result = [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence]
    result.append(EOS_token)
    return result


def indexesFromPair(lang, pair):
    input = indexesFromSentence(lang, pair[0])
    target = indexesFromSentence(lang, pair[1])
    return input, target


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
        self.gru = nn.GRU(hidden_size * 2 + embed_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.gen_out = nn.Linear(hidden_size, dec_vocab_size)
        self.copy_out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_id, input, encoder_outputs, encoder_input_ids, hidden, attention):
        '''
        :param input_id: (b, 1)
        :param input: (b, 1, embed)
        :param encoder_outputs: (b, l, hidden)
        :param encoder_input_ids: (b, l)
        :param hidden: (b, 1, hidden)
        :param attention: (b, 1, hidden)
        :return:
        '''
        b = encoder_input_ids.size(0)
        l = encoder_input_ids.size(1)
        mask = torch.eq(input_id.expand(-1, l), encoder_input_ids).float()  # (b, l)
        mask_sum = torch.sum(mask, 1, keepdim=True)  # (b, 1)
        rou = mask / mask_sum  # (b, l)
        rou[torch.isnan(rou)] = 0  # (b, l)
        selective_read = torch.bmm(rou.unsqueeze(1), encoder_outputs)  # (b, 1, hidden_size)

        input = self.dropout(input)  # (b, 1, embed)

        input = torch.cat((input, attention, selective_read), 2)  # (b, 1, hidden + embed)
        _, cur_hidden = self.gru(input, hidden.transpose(0, 1))
        cur_hidden = cur_hidden.transpose(0, 1)  # (b, 1, hidden)

        attn_weights = F.softmax(torch.bmm(self.attn(cur_hidden),
                                           encoder_outputs.transpose(1, 2)), dim=2)  # (b, 1, l)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # (b, 1, hidden)
        cur_attention = torch.tanh(
            self.attn_combine(torch.cat((attn_applied, cur_hidden), 2)))  # (b, 1, hidden_size)

        generate_score = self.gen_out(cur_attention).squeeze(1)  # (b, dec_vocab_size)

        # CopyNet
        copy_weights = torch.sigmoid(self.copy_out(encoder_outputs))  # (b, l, hidden)
        copy_score = torch.bmm(copy_weights, cur_attention.transpose(1, 2)).squeeze(2)  # (b, l)

        score = F.softmax(torch.cat((generate_score, copy_score), 1), dim=1)
        generate_score, copy_score = torch.split(score, (self.dec_vocab_size, l), 1)

        prob_g = torch.cat((generate_score, torch.zeros(b, self.vocab_size - self.dec_vocab_size, device=device)), 1)  # (b, vocab_size)
        
        # scatter_add_ 是0.5中的函数，0.4中还未发布
        # prob_c = torch.zeros(b, self.vocab_size, device=device).scatter_add_(1, encoder_input_ids, copy_score)  # (b, vocab_size)
        prob_c = torch.zeros(b, self.vocab_size, device=device)
        for b_idx in range(b):
            for l_idx in range(l):
                prob_c[b_idx, encoder_input_ids[b_idx, l_idx]] += copy_score[b_idx, l_idx]


        output = prob_g + prob_c  # (b, vocab_size)
        output[output == 0] = float('-inf')  # 将概率为0的项替换成-inf
        output = torch.log(output)  # (b, vocab_size)
        output[torch.isnan(output)] = float('-inf')  # 将nan替换成-inf

        # output = F.log_softmax(generate_score, dim=1)  # (b, dec_vocab_size)
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

    decoder_input_id = torch.zeros(b, 1, dtype=torch.long, device=device).fill_(SOS_token)  # (b, 1)
    decoder_input = embedding(decoder_input_id)  # (b, 1, embed)
    decoder_attention = torch.zeros(b, 1, decoder.hidden_size, device=device)  # (b, 1, hidden)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(len(target[0])):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input_id, decoder_input,
                                                                        encoder_outputs, input, decoder_hidden,
                                                                        decoder_attention)
            loss += criterion(decoder_output, target[:, di])

            decoder_input_id = target[:, di].view(-1, 1)  # (b, 1)
            decoder_input = embedding(decoder_input_id)  # (b, 1, embed)

    loss.backward()
    # print(map(lambda x: x.grad, embedding.parameters()))
    # print(map(lambda x: x.grad, encoder.parameters()))
    # print(map(lambda x: x.grad, decoder.parameters()))
    clip_grad_norm_(embedding.parameters(), 5)
    clip_grad_norm_(encoder.parameters(), 5)
    clip_grad_norm_(decoder.parameters(), 5)

    optimizer.step()

    return loss.item() / len(target_lens)


def trainIters(embedding, encoder, decoder, train_pairs, max_length, n_iters, learning_rate, batch_size, print_every=1000, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    train_data = []
    best_bleu_score = 0.0
    optimizer = optim.Adam([{"params": embedding.parameters()}, {"params": encoder.parameters()},
                            {"params": decoder.parameters()}], lr=learning_rate, amsgrad=True)
    for pair in train_pairs:
        train_data.append(indexesFromPair(lang, pair))
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        print("======================iter%s============================" % iter)
        for i, training_pairs in enumerate(get_minibatches(train_data, batch_size)):
            print("batch: ", i)
            # 排序并padding
            training_pairs = sorted(training_pairs, key=lambda x: len(x[0]), reverse=True)
            enc_lens = list(map(lambda x: len(x[0]), training_pairs))
            dec_lens = list(map(lambda x: len(x[1]), training_pairs))
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

        bleu_score, _ = evaluate(embedding, encoder, decoder, dev_pairs, max_length)
        bleu_score = bleu_score if bleu_score is not None else 0
        print('bleu_socre: {0}'.format(bleu_score))
        if bleu_score >= best_bleu_score:
            best_bleu_score = bleu_score
            torch.save(encoder.state_dict(), 'model/encoder')
            torch.save(decoder.state_dict(), 'model/decoder')
            torch.save(embedding.state_dict(), 'model/embedding')
            print('new model saved.')


class Node(object):
    def __init__(self, parent, state, value, cost, extras):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent # parent Node, None for root
        self.state = state if state is not None else None # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras # can hold, for example, attention weights
        self._sequence = None

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [s.value for s in self.to_sequence()]

    def to_sequence_of_extras(self):
        return [s.extras for s in self.to_sequence()]


def beam_search(encoder_outputs, decoder_hidden, decoder_attention, decoder, embedding, input, max_length,
                start_id=SOS_token, end_id=EOS_token, beam_width=2, num_hypotheses=1):
    next_fringe = [Node(parent=None, state=decoder_hidden, value=start_id, cost=0.0, extras=None)]
    hypotheses = []

    for _ in range(max_length):

        fringe = []
        for n in next_fringe:
            if n.value == end_id:
                hypotheses.append(n)
            else:
                fringe.append(n)

        if not fringe or len(hypotheses) >= num_hypotheses:
            break

        decoder_input_ids = [n.value for n in fringe]
        decoder_hiddens = torch.cat([n.state for n in fringe])

        Y_t = []
        p_t = []
        extras_t = []
        state_t = []
        for i in range(len(decoder_input_ids)):
            decoder_input_id = torch.tensor([[decoder_input_ids[i]]], dtype=torch.long, device=device)
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input_id, embedding(decoder_input_id),
                                                                                    encoder_outputs, input, decoder_hiddens[i].view(1,1,-1),
                                                                                    decoder_attention)
            topv, topi = decoder_output.data.topk(beam_width)
            Y_t.append(topi.tolist()[0])
            p_t.append(F.softmax(decoder_output, dim=0))
            state_t.append(decoder_hidden)
            extras_t.append(decoder_attention)

        next_fringe = []
        for Y_t_n, p_t_n, extras_t_n, state_t_n, n in zip(Y_t, p_t, extras_t, state_t, fringe):
            Y_nll_t_n = -np.log(p_t_n[0][Y_t_n])
            for y_t_n, y_nll_t_n in zip(Y_t_n, Y_nll_t_n):
                n_new = Node(parent=n, state=state_t_n, value=y_t_n, cost=y_nll_t_n, extras=extras_t_n)
                next_fringe.append(n_new)
        next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost)[:beam_width] # may move this into loop to save memory

    hypotheses.sort(key=lambda n: n.cum_cost)
    return hypotheses[:num_hypotheses]


def evaluate(embedding, encoder, decoder, dev_pairs, max_length, bms=False):
    with torch.no_grad():
        output_sentences = []
        # Inference mode (disable dropout)
        encoder.eval()
        decoder.eval()

        for pair in dev_pairs:
            input = torch.tensor([indexesFromSentence(lang, pair[0])], dtype=torch.long, device=device)

            encoder_input = embedding(input)  # (1, l, embed)
            encoder_outputs, decoder_hidden = encoder(encoder_input)  # (1, l, hidden), (1, 1, hidden)

            decoder_input_id = torch.zeros(1, 1, dtype=torch.long, device=device).fill_(SOS_token)  # (1, 1)
            decoder_input = embedding(decoder_input_id)  # (1, 1, embed)
            decoder_attention = torch.zeros(1, 1, decoder.hidden_size, device=device)  # (1, 1, hidden)

            decoded_words = []

            if not bms:
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input_id, decoder_input,
                                                                                encoder_outputs, input, decoder_hidden,
                                                                                decoder_attention)
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(lang.index2word[topi.item()])

                    decoder_input_id = topi.squeeze().detach().view(1, 1)  # (1, 1)
                    decoder_input = embedding(decoder_input_id)  # (1, 1, embed).
            else:
                hypotheses = beam_search(encoder_outputs, decoder_hidden, decoder_attention, decoder, embedding, input,
                                         max_length)
                for hypothesis in hypotheses:
                    decoder_indices = hypothesis.to_sequence_of_values()
                    decoded_words.append([lang.index2word[i] for i in decoder_indices][1:-1])
                    decoded_words = decoded_words[0]

            output_sentences.append(' '.join(decoded_words).replace('<EOS>', ''))

        with open('bleu/predict', 'w', encoding='utf-8') as pr:
            pr.write('\n'.join(output_sentences))
        p = subprocess.Popen(['perl', 'bleu/multi-bleu.pl', 'bleu/gold'], stdin=open('bleu/predict'),
                             stdout=subprocess.PIPE)
        lines = p.stdout.readlines()
        return float(lines[0].split()[0][:-1]), output_sentences


def test(embedding, encoder, decoder, test_pairs, max_length):
    for pair in test_pairs:
        print('>', pair[0])
        print('=', pair[1])
        _, output_sentences = evaluate(embedding, encoder, decoder, [pair], max_length)
        print('<', output_sentences[0])
        print('')


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
    files['trainAnswers'].extend(open('data/answers-' + str(i) + '.txt', encoding="utf-8").read().strip().split('\n'))
    files['trainQuestions'].extend(open('data/questions-' + str(i) + '.txt', encoding="utf-8").read().strip().split('\n'))
files['devAnswers'].extend(open('data/answers-' + str(dev_id) + '.txt', encoding="utf-8").read().strip().split('\n'))
files['devQuestions'].extend(open('data/questions-' + str(dev_id) + '.txt', encoding="utf-8").read().strip().split('\n'))

# for debug
'''
files['trainAnswers'] = files['trainAnswers'][:5]
files['trainQuestions'] = files['trainQuestions'][:5]
files['devAnswers'] = files['trainAnswers']
files['devQuestions'] = files['trainQuestions']
'''

# 生成词表
lang = prepare_vocabulary(word_tokenizer, files, cut=3)
print('dec_vocab_size: ', lang.n_words_for_decoder)
print('vocab_size: ', lang.n_words)
print('max_word_length: ', max(map(lambda x: len(x), lang.word2index)))

# 生成数据
data = prepare_data(word_tokenizer, files)
train_pairs = []
dev_pairs = []
length = []
for i in range(len(data['trainQuestions'])):
    train_pairs.append((data['trainQuestions'][i], data['trainAnswers'][i]))
    length.append(len(data['trainAnswers'][i]))
for i in range(len(data['devQuestions'])):
    dev_pairs.append((data['devQuestions'][i], data['devAnswers'][i]))
    length.append(len(data['devAnswers'][i]))
print('max_output_length: ', max(length))

# 实际的序列会多一个终止token
max_length = max(length) + 1

n_epochs = 100
batch_size = 32
lr = 0.001
embed_size = 200
hidden_size = 256

# 共用一套embedding
embedding = nn.Embedding(lang.n_words, embed_size, padding_idx=0).to(device)
encoder = EncoderRNN(embed_size, lang.n_words, hidden_size // 2).to(device)
attn_decoder = CopynetDecoderRNN(embed_size, hidden_size, lang.n_words_for_decoder, lang.n_words, dropout_p=0.1).to(
    device)

trainIters(embedding, encoder, attn_decoder, train_pairs, max_length, n_epochs, lr, batch_size, print_every=1)
