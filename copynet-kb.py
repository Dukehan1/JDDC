# -*- coding: UTF-8 -*-
import numpy as np
import re
import random
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm_
import os
import pickle
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from dataToOpenNMT import word_tokenizer, sentence_with_kb

device_cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3


VECTOR_DIR = 'embed/sgns.merge.word'


def init_embedding(embed_size, n_word, word2index):
    if os.path.exists('model_copynet/pretrained'):
        t = open('model_copynet/pretrained', 'rb')
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

        t = open('model_copynet/pretrained', 'wb')
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
    result.append(EOS_token)
    return result


def indexesFromPair(lang, pair):
    input = indexesFromSentence(lang, pair[0])
    target = indexesFromSentence(lang, pair[1])
    return input, target


class EncoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, bidirectional=True, num_layers=2, dropout_p=0.5):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.lstm = nn.LSTM(embed_size, hidden_size // 2, num_layers=self.num_layers, batch_first=True,
                                bidirectional=True, dropout=self.dropout_p)
        else:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=self.num_layers, batch_first=True,
                                bidirectional=False, dropout=self.dropout_p)

    def forward(self, input):
        '''
        :param input: (b, l, embed)
        :return:
        '''
        # bidirectional = True: (b, l, hidden), (num_layers * 2, b, hidden / 2)
        # bidirectional = False: (b, l, hidden), (num_layers, b, hidden)
        output, (h_n, c_n) = self.lstm(input)
        if self.bidirectional:
            h_n = h_n.transpose(0, 1)
            h_n = h_n.reshape(-1, self.num_layers, self.hidden_size)
            h_n = h_n.transpose(0, 1)  # (num_layers, b, hidden)
            c_n = c_n.transpose(0, 1)
            c_n = c_n.reshape(-1, self.num_layers, self.hidden_size)
            c_n = c_n.transpose(0, 1)  # (num_layers, b, hidden)

        return output, (h_n, c_n)

    def pack_unpack(self, input, input_lens):
        packed_input = pack_padded_sequence(input, input_lens, batch_first=True)
        packed_output, (h_n, c_n) = self.forward(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, (h_n, c_n)


class CopynetDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, dec_vocab_size, vocab_size, num_layers=2, dropout_p=0.5):
        super(CopynetDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.vocab_size = vocab_size
        self.dec_vocab_size = dec_vocab_size

        self.lstm = nn.LSTM(hidden_size * 2 + embed_size, hidden_size, num_layers=self.num_layers, batch_first=True,
                           dropout=self.dropout_p)
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
        :param hidden: (num_layers, b, hidden)
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

        encoder_input_ids_mask = encoder_input_ids == 0

        input = torch.cat((input, attention, selective_read), 2)  # (b, 1, hidden * 2 + embed)
        o, cur_hidden = self.lstm(input, hidden)  # (b, 1, hidden), (num_layers, b, hidden)

        attn_weights = F.softmax(torch.bmm(self.attn(o),
                                           encoder_outputs.transpose(1, 2)).masked_fill_(
            encoder_input_ids_mask.unsqueeze(1), -float('inf')), dim=2)  # (b, 1, l)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # (b, 1, hidden)
        cur_attention = torch.tanh(
            self.attn_combine(torch.cat((attn_applied, o), 2)))  # (b, 1, hidden)

        generate_score = self.gen_out(cur_attention).squeeze(1)  # (b, dec_vocab_size)

        # CopyNet
        copy_weights = torch.sigmoid(self.copy_out(encoder_outputs))  # (b, l, hidden)
        copy_score = torch.bmm(copy_weights, cur_attention.transpose(1, 2)).squeeze(2).masked_fill_(
            encoder_input_ids_mask, -float('inf'))  # (b, l)

        score = F.softmax(torch.cat((generate_score, copy_score), 1), dim=1)
        generate_score, copy_score = torch.split(score, (self.dec_vocab_size, l), 1)

        prob_g = torch.cat((generate_score, torch.zeros(b, self.vocab_size - self.dec_vocab_size, device=device_cuda)), 1)  # (b, vocab_size)
        
        # scatter_add_ 是0.4.1中的函数
        prob_c = torch.zeros(b, self.vocab_size, device=device_cuda).scatter_add_(1, encoder_input_ids, copy_score)  # (b, vocab_size)

        output = prob_g + prob_c  # (b, vocab_size)
        output.masked_fill_(output == 0, float('-inf'))  # 将概率为0的项替换成-inf
        output = torch.log(output)  # (b, vocab_size)
        output.masked_fill_(torch.isnan(output), float('-inf'))  # 将nan替换成-inf

        # output = F.log_softmax(generate_score, dim=1)  # (b, vocab_size)
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


class Node(object):
    def __init__(self, decoder_input_id, decoder_input, decoder_hidden, decoder_attention, parent, cost):
        super(Node, self).__init__()
        self.decoder_input_id = decoder_input_id
        self.decoder_input = decoder_input
        self.decoder_hidden = decoder_hidden
        self.decoder_attention = decoder_attention
        self.parent = parent # parent Node, None for root
        self.cost = cost
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
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
        return [s.decoder_input_id.item() for s in self.to_sequence()]


def beam_search(embedding, decoder, decoder_input_id, decoder_input, encoder_outputs, encoder_input_ids, decoder_hidden, decoder_attention,
                max_length, start_id=SOS_token, end_id=EOS_token, beam_width=3, num_hypotheses=1):
    '''
    # 不支持batch
    :param decoder_input_id: (b, 1)
    :param decoder_input: (b, 1, embed)
    :param encoder_outputs: (b, l, hidden)
    :param encoder_input_ids: (b, l)
    :param decoder_hidden: (num_layers, b, hidden)
    :param decoder_attention: (b, 1, hidden)
    :return:
    '''
    next_fringe = [Node(decoder_input_id, decoder_input, decoder_hidden, decoder_attention, None, 0.0)]
    hypotheses = []

    for _ in range(max_length):

        fringe = []
        for n in next_fringe:
            if n.decoder_input_id.item() == end_id:
                hypotheses.append(n)
            else:
                fringe.append(n)

        if not fringe or len(hypotheses) >= num_hypotheses:
            break

        decoder_input_ids = [n.decoder_input_id for n in fringe]
        decoder_inputs = [n.decoder_input for n in fringe]
        decoder_hiddens = [n.decoder_hidden for n in fringe]
        decoder_attentions = [n.decoder_attention for n in fringe]

        Y_t = []
        p_t = []
        decoder_hidden_t = []
        decoder_attention_t = []
        for i in range(len(fringe)):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input_ids[i], decoder_inputs[i],
                                                                        encoder_outputs, encoder_input_ids,
                                                                        decoder_hiddens[i], decoder_attentions[i])
            topv, topi = decoder_output.topk(beam_width)
            Y_t.append(topi)
            p_t.append(decoder_output)
            decoder_hidden_t.append(decoder_hidden)
            decoder_attention_t.append(decoder_attention)

        next_fringe = []
        for Y_t_n, p_t_n, decoder_hidden_t_n, decoder_attention_t_n, n in zip(Y_t, p_t, decoder_hidden_t, decoder_attention_t, fringe):
            Y_nll_t_n = -p_t_n[:, Y_t_n[0]]
            for y_t_n, y_nll_t_n in zip(Y_t_n.squeeze(), Y_nll_t_n.squeeze()):
                n_new = Node(y_t_n.view(1, 1), embedding(y_t_n.view(1, 1)), decoder_hidden_t_n, decoder_attention_t_n, n, y_nll_t_n.item())
                next_fringe.append(n_new)
        next_fringe = sorted(next_fringe, key=lambda x: x.cost)[:beam_width]

    if not hypotheses:
        hypotheses = next_fringe
    hypotheses.sort(key=lambda x: x.cum_cost)
    return hypotheses[:num_hypotheses]


teacher_forcing_ratio = 1.0


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


def train(input, input_lens, target, target_lens, embedding, encoder, decoder, optimizer, criterion):
    # Training mode (enable dropout)
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    loss = 0

    b = len(input)

    encoder_input = embedding(input)  # (b, l, embed)
    encoder_outputs, decoder_hidden = encoder.module.pack_unpack(encoder_input, input_lens)  # (b, l, hidden), (num_layers, b, hidden)

    decoder_input_id = torch.zeros(b, 1, dtype=torch.long, device=device_cuda).fill_(SOS_token)  # (b, 1)
    decoder_input = embedding(decoder_input_id)  # (b, 1, embed)
    decoder_attention = torch.zeros(b, 1, decoder.module.hidden_size, device=device_cuda)  # (b, 1, hidden)

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
    clip_grad_norm_(embedding.parameters(), 2)
    clip_grad_norm_(encoder.parameters(), 2)
    clip_grad_norm_(decoder.parameters(), 2)

    optimizer.step()

    return loss.data.item() / len(target[0])


def inference(lang, input, input_lens, embedding, encoder, decoder, max_length, bms=False):
    with torch.no_grad():
        # Inference mode (disable dropout)
        encoder.eval()
        decoder.eval()

        b = len(input)

        encoder_input = embedding(input)  # (b, l, embed)
        encoder_outputs, decoder_hidden = encoder.module.pack_unpack(encoder_input, input_lens)  # (b, l, hidden), (num_layers, b, hidden)

        decoder_input_id = torch.zeros(b, 1, dtype=torch.long, device=device_cuda).fill_(SOS_token)  # (b, 1)
        decoder_input = embedding(decoder_input_id)  # (b, 1, embed)
        decoder_attention = torch.zeros(b, 1, decoder.module.hidden_size, device=device_cuda)  # (b, 1, hidden)

        decoded_idxs = np.empty((b, 0), dtype=int)
        if not bms:
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input_id, decoder_input,
                                                                                encoder_outputs, input, decoder_hidden,
                                                                                decoder_attention)
                topi = torch.argmax(decoder_output, dim=1, keepdim=True)  # (b, 1)
                decoded_idxs = np.concatenate((decoded_idxs, topi.data.cpu().numpy()), axis=1)
                    
                decoder_input_id = topi.detach()  # (b, 1)
                decoder_input = embedding(decoder_input_id)  # (b, 1, embed)
        else:
            # 不支持batch
            hypotheses = beam_search(embedding, decoder, decoder_input_id, decoder_input, encoder_outputs, input,
                                     decoder_hidden, decoder_attention, max_length)
            decoded_idxs = [hypotheses[0].to_sequence_of_values()[1:]]

        decoded_words = []
        for s in decoded_idxs:
            sentence = []
            for w in s:
                if w != EOS_token:
                    # 模型中' '会被当成''，这一步需要再替换回来
                    temp = lang.index2word[w] if lang.index2word[w] != '' else ' '
                    sentence.append(temp)
                else:
                    break
            decoded_words.append(sentence)

        return decoded_words


def trainIters(lang, embedding, encoder, decoder, optimizer, train_pairs, dev_pairs, max_length, n_iters, learning_rate, batch_size, infer_batch_size, print_every=1000, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_bleu_score = 0.0
    criterion = nn.NLLLoss(ignore_index=0)

    # 在有Model时预先生成best_bleu_score
    print("============evaluate_start==================")
    bleu_score = evaluate(lang, embedding, encoder, decoder, dev_pairs, max_length, infer_batch_size)
    print('bleu_socre: {0}'.format(bleu_score))
    if bleu_score >= best_bleu_score:
        best_bleu_score = bleu_score
        torch.save(encoder.state_dict(), 'model_copynet/encoder')
        torch.save(decoder.state_dict(), 'model_copynet/decoder')
        torch.save(embedding.state_dict(), 'model_copynet/embedding')
        torch.save(optimizer.state_dict(), 'model_copynet/optimizer')
        print('new model_copynet saved.')
    print("==============evaluate_end==================")

    for iter in range(1, n_iters + 1):
        print("======================iter%s============================" % iter)
        for i, training_batch in enumerate(get_minibatches(train_pairs, batch_size)):

            # print("batch: ", i)
            # 排序并padding
            training_batch = sorted(training_batch, key=lambda x: len(x[0]), reverse=True)
            training_batch = list(map(lambda x: indexesFromPair(lang, x), training_batch))
            enc_lens = list(map(lambda x: len(x[0]), training_batch))
            dec_lens = list(map(lambda x: len(x[1]), training_batch))
            # print("enc_lens: ", enc_lens)
            # print("dec_lens: ", dec_lens)
            enc_max_len = max(enc_lens)
            dec_max_len = max(dec_lens)
            enc = []
            dec = []
            for t in training_batch:
                enc.append(t[0] + (enc_max_len - len(t[0])) * [0])
                dec.append(t[1] + (dec_max_len - len(t[1])) * [0])
            enc = torch.tensor(enc, dtype=torch.long, device=device_cuda)
            dec = torch.tensor(dec, dtype=torch.long, device=device_cuda)

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

        print("============evaluate_start==================")
        bleu_score = evaluate(lang, embedding, encoder, decoder, dev_pairs, max_length, infer_batch_size)
        print('bleu_socre: {0}'.format(bleu_score))
        if bleu_score >= best_bleu_score:
            best_bleu_score = bleu_score
            torch.save(encoder.state_dict(), 'model_copynet/encoder')
            torch.save(decoder.state_dict(), 'model_copynet/decoder')
            torch.save(embedding.state_dict(), 'model_copynet/embedding')
            torch.save(optimizer.state_dict(), 'model_copynet/optimizer')
            print('new model_copynet saved.')
        print("==============evaluate_end==================")


def evaluate(lang, embedding, encoder, decoder, dev_pairs, max_length, infer_batch_size):
    target_words = []
    decoded_words = []
    for i in range(0, len(dev_pairs), infer_batch_size):
        # print("batch: ", int(i / infer_batch_size))
        dev_batch = dev_pairs[i:i + infer_batch_size]
        # 排序并padding
        dev_batch = sorted(dev_batch, key=lambda x: len(x[0]), reverse=True)
        target_words.extend(map(lambda x: x[1], dev_batch))
        dev_batch = list(map(lambda x: indexesFromPair(lang, x), dev_batch))
        enc_lens = list(map(lambda x: len(x[0]), dev_batch))
        # print("enc_lens: ", enc_lens)
        enc_max_len = max(enc_lens)
        enc = []
        for t in dev_batch:
            enc.append(t[0] + (enc_max_len - len(t[0])) * [0])
        enc = torch.tensor(enc, dtype=torch.long, device=device_cuda)

        decoded_words.extend(inference(lang, enc, enc_lens, embedding, encoder, decoder, max_length))

    target_sentences = list(map(lambda x: ''.join(x), target_words))
    output_sentences = list(map(lambda x: ''.join(x), decoded_words))

    with open('bleu/gold', 'w', encoding='utf-8') as gw:
        gw.write('\n'.join(target_sentences))
        gw.close()
    with open('bleu/predict', 'w', encoding='utf-8') as pr:
        pr.write('\n'.join(output_sentences))
        pr.close()

    bleu_score = bleu(target_sentences, output_sentences)
    return bleu_score

# 实际的序列会多一个终止token
max_seq_length = 150
max_num_utterance = 8
DEC_MAX_LEN = 150
max_length = DEC_MAX_LEN + 1

n_epochs = 40
lr = 0.001
batch_size = 16
infer_batch_size = 512
embed_size = 300
hidden_size = 256
bidirectional = True
dropout_p = 0.5
num_layers = 2
cut = 8


def cut_utterances(chunk):
    '''
    裁剪并填充utterances
    '''
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
    temp = utterances[1:]
    temp.reverse()
    utterances = utterances[:1] + temp
    utterances = utterances + (max_num_utterance - len(utterances)) * [[]]
    utterances = list(map(lambda x: x[:max_length], utterances))
    utterances = list(map(lambda x: x + (max_length - len(x)) * ['<PAD>'], utterances))
    result = []
    for u in utterances:
        result.extend(u)
    return result


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
    '''
    data['trainAnswers'] = data['trainAnswers'][:2]
    data['trainQuestions'] = data['trainQuestions'][:2]
    data['devAnswers'] = [list(x) for x in data['trainAnswers']]
    data['devQuestions'] = [list(x) for x in data['trainQuestions']]
    '''

    # 生成词表（word）
    if os.path.exists('model_copynet/vocab_word'):
        t = open('model_copynet/vocab_word', 'rb')
        vocab_word = pickle.load(t)
        t.close()
    else:
        vocab_word = prepare_vocabulary(data, cut=cut)
        t = open('model_copynet/vocab_word', 'wb')
        pickle.dump(vocab_word, t)
        t.close()
    print("========================word===========================")
    print('dec_vocab_size: ', vocab_word.n_words_for_decoder)
    print('vocab_size: ', vocab_word.n_words)
    print('max_word_length: ', max(map(lambda x: len(x), vocab_word.word2index)))

    # 生成数据（截断）
    if os.path.exists('model_copynet/data'):
        t = open('model_copynet/data', 'rb')
        train_pairs, dev_pairs = pickle.load(t)
        t.close()
    else:
        train_pairs = []
        dev_pairs = []
        for i in range(len(data['trainQuestions'])):
            data['trainQuestions'][i] = cut_utterances(data['trainQuestions'][i])
            data['trainAnswers'][i] = data['trainAnswers'][i][:DEC_MAX_LEN]
            train_pairs.append((data['trainQuestions'][i], data['trainAnswers'][i]))
        for i in range(len(data['devQuestions'])):
            data['devQuestions'][i] = cut_utterances(data['devQuestions'][i])
            data['devAnswers'][i] = data['devAnswers'][i][:DEC_MAX_LEN]
            dev_pairs.append((data['devQuestions'][i], data['devAnswers'][i]))
        t = open('model_copynet/data', 'wb')
        pickle.dump((train_pairs, dev_pairs), t)
        t.close()

    # 共用一套embedding
    embed = init_embedding(embed_size, vocab_word.n_words, vocab_word.word2index)
    embedding = nn.Embedding(vocab_word.n_words, embed_size, padding_idx=0).from_pretrained(embed, freeze=False)
    encoder = EncoderRNN(embed_size, vocab_word.n_words, hidden_size, bidirectional=bidirectional,
                         num_layers=num_layers, dropout_p=dropout_p)
    attn_decoder = CopynetDecoderRNN(embed_size, hidden_size, vocab_word.n_words_for_decoder, vocab_word.n_words,
                                     num_layers=num_layers, dropout_p=dropout_p)

    embedding = torch.nn.DataParallel(embedding).to(device_cuda)
    encoder = torch.nn.DataParallel(encoder).to(device_cuda)
    attn_decoder = torch.nn.DataParallel(attn_decoder).to(device_cuda)

    if os.path.isfile('model_copynet/embedding'):
        embedding.load_state_dict(torch.load('model_copynet/embedding'))

    if os.path.isfile('model_copynet/encoder') and os.path.isfile('model_copynet/decoder'):
        encoder.load_state_dict(torch.load('model_copynet/encoder'))
        attn_decoder.load_state_dict(torch.load('model_copynet/decoder'))

    optimizer = optim.Adam([{"params": embedding.parameters()}, {"params": encoder.parameters()},
                            {"params": attn_decoder.parameters()}], lr=lr, amsgrad=True)

    if os.path.isfile('model_copynet/optimizer'):
        optimizer.load_state_dict(torch.load('model_copynet/optimizer'))

    trainIters(vocab_word, embedding, encoder, attn_decoder, optimizer, train_pairs, dev_pairs, max_length, n_epochs, lr,
               batch_size, infer_batch_size, print_every=1)


def run_prediction(input_file_path, output_file_path):
    # 生成词表（word）
    t = open('model_copynet/vocab_word', 'rb')
    vocab_word = pickle.load(t)
    t.close()
    print("========================word===========================")
    print('dec_vocab_size: ', vocab_word.n_words_for_decoder)
    print('vocab_size: ', vocab_word.n_words)
    print('max_word_length: ', max(map(lambda x: len(x), vocab_word.word2index)))

    # 共用一套embedding
    embedding = nn.Embedding(vocab_word.n_words, embed_size, padding_idx=0)
    encoder = EncoderRNN(embed_size, vocab_word.n_words, hidden_size, bidirectional=bidirectional,
                         num_layers=num_layers, dropout_p=dropout_p)
    attn_decoder = CopynetDecoderRNN(embed_size, hidden_size, vocab_word.n_words_for_decoder, vocab_word.n_words,
                                     num_layers=num_layers, dropout_p=dropout_p)

    embedding = torch.nn.DataParallel(embedding).to(device_cuda)
    encoder = torch.nn.DataParallel(encoder).to(device_cuda)
    attn_decoder = torch.nn.DataParallel(attn_decoder).to(device_cuda)

    embedding.load_state_dict(torch.load('model_copynet/embedding', map_location='cpu'))
    encoder.load_state_dict(torch.load('model_copynet/encoder', map_location='cpu'))
    attn_decoder.load_state_dict(torch.load('model_copynet/decoder', map_location='cpu'))

    input_data = open(input_file_path, encoding="utf-8").read().strip().split('\n')
    print(input_data)
    # 添加KB
    input_data = list(map(lambda x: sentence_with_kb(x), input_data))
    # 分词
    input_data = list(map(lambda x: word_tokenizer(x), input_data))
    print(input_data)

    decoded_words = []
    for s in input_data:
        s = indexesFromSentence(vocab_word, s)
        enc = [s]
        enc = torch.tensor(enc, dtype=torch.long, device=device_cuda)
        enc_lens = [len(s)]
        result = inference(vocab_word, enc, enc_lens, embedding, encoder, attn_decoder, max_length, bms=True)
        print(result)
        decoded_words.extend(result)

    # 后处理，去掉重复空格以及<UNK>
    output_sentences = list(map(lambda x: re.sub('(\s+|<UNK>)', ' ', ''.join(x)), decoded_words))

    output_file = open(output_file_path, 'w', encoding='utf-8')
    output_file.write('\n'.join(output_sentences))

if __name__ == "__main__":
    run_train()
