# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
from torch.nn.utils import clip_grad_norm_
import os
import pickle
import math

from utils import init_embedding, device_cuda, gen_data, prepare_vocabulary, ComputeR10_1, get_minibatches, \
    sentence_with_kb, word_tokenizer


class SMN(nn.Module):
    def __init__(self, embed_size, hidden_size, compression_hidden_size, max_num_utterance, max_length,
                 bidirectional, num_layers, dropout_p):
        super(SMN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.compression_hidden_size = compression_hidden_size
        self.max_num_utterance = max_num_utterance
        self.max_length = max_length
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(self.dropout_p)

        if self.bidirectional:
            self.sentence_gru = nn.LSTM(embed_size, hidden_size // 2, bidirectional=True, num_layers=num_layers, batch_first=True)
            self.final_gru = nn.LSTM(compression_hidden_size, compression_hidden_size // 2, bidirectional=True, batch_first=True)
        else:
            self.sentence_gru = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.final_gru = nn.LSTM(compression_hidden_size, compression_hidden_size, batch_first=True)

        self.a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        )
        self.matching = nn.Sequential(
            nn.Linear(16 * pow(math.floor((max_length - 5) / 3 + 1), 2), compression_hidden_size),
            nn.ReLU()
        )
        self.final = nn.Linear(compression_hidden_size, 2)

    def forward(self, utterances, response):
        '''
        :param utterances: (b, max_num_utterance, l, embed)
        :param response: (b, l, embed)
        :return:
        '''
        b = response.size(0)
        utterances = utterances.transpose(0, 1)
        matching_vectors = []
        response_output, _ = self.sentence_gru(response)  # (b, l, hidden)
        response_output = self.dropout(response_output)
        for utterance in utterances:
            M1 = torch.bmm(utterance, response.transpose(1, 2))  # (b, l, l)

            utterance_output, _ = self.sentence_gru(utterance)  # (b, l, hidden)
            utterance_output = self.dropout(utterance_output)
            utterance_output_trans = self.a(utterance_output)  # (b, l, hidden)
            M2 = torch.bmm(utterance_output_trans, response_output.transpose(1, 2))  # (b, l, l)

            matching_vector = torch.stack([M1, M2], 1)  # (b, 2, l, l)
            matching_vector = self.conv(matching_vector)
            matching_vector = self.matching(matching_vector.view(b, -1))  # (b, 50)
            matching_vectors.append(matching_vector.view(b, -1))
        matching_vectors = torch.stack(matching_vectors, 1)  # (b, max_num_utterance, 50)
        _, (hidden, _) = self.final_gru(matching_vectors)  # (1, b, 50)
        hidden = hidden.transpose(0, 1).reshape(b, -1)
        output = F.log_softmax(self.final(hidden), dim=1)  # (b, 2)
        output_prob = F.softmax(self.final(hidden), dim=1)  # (b, 2)
        return output, output_prob


def train(utterances, response, label, embedding, smn, optimizer, criterion):
    # Training mode (enable dropout)
    smn.train()
    optimizer.zero_grad()

    utterances = embedding(utterances)
    response = embedding(response)
    output, _ = smn(utterances, response)
    loss = criterion(output, label)

    loss.backward()
    # print(list(map(lambda x: x.grad, embedding.parameters())))
    # print(list(map(lambda x: x.grad, smn.parameters())))
    clip_grad_norm_(embedding.parameters(), 2)
    clip_grad_norm_(smn.parameters(), 2)

    optimizer.step()

    return loss.data.item()


def inference(utterances, response, embedding, smn):
    with torch.no_grad():
        # Inference mode (disable dropout)
        smn.eval()

        utterances = embedding(utterances)
        response = embedding(response)
        _, output = smn(utterances, response)
        output = output[:, 1]
        output = output.data.cpu().numpy()

        return output


def trainIters(vocab, embedding, smn, optimizer, train_examples, dev_examples, n_iters, learning_rate, batch_size,
               infer_batch_size, print_every=1000, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_score = 0.0
    criterion = nn.NLLLoss()

    # 在有Model时预先生成best_score
    print("============evaluate_start==================")
    score = evaluate(embedding, smn, dev_examples, infer_batch_size)
    print('R10@1_socre: {0}'.format(score))
    if score >= best_score:
        best_score = score
        torch.save(smn.state_dict(), model_dir + '/smn')
        torch.save(embedding.state_dict(), model_dir + '/embedding')
        torch.save(optimizer.state_dict(), model_dir + '/optimizer')
        print('new ' + model_dir + ' saved.')
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

            loss = train(utterances, response, label, embedding, smn, optimizer, criterion)

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
        score = evaluate(embedding, smn, dev_examples, infer_batch_size)
        print('R10@1_socre: {0}'.format(score))
        if score >= best_score:
            best_score = score
            torch.save(smn.state_dict(), model_dir + '/smn')
            torch.save(embedding.state_dict(), model_dir + '/embedding')
            torch.save(optimizer.state_dict(), model_dir + '/optimizer')
            print('new ' + model_dir + ' saved.')
        print("==============evaluate_end==================")


def evaluate(embedding, smn, dev_examples, infer_batch_size):
    predict = []
    for i in range(0, len(dev_examples[0]), infer_batch_size):
        # print("batch: ", int(i / infer_batch_size))
        utterances = dev_examples[0][i:i + infer_batch_size]
        response = dev_examples[1][i:i + infer_batch_size]

        utterances = torch.tensor(utterances, dtype=torch.long, device=device_cuda)
        response = torch.tensor(response, dtype=torch.long, device=device_cuda)

        predict.extend(inference(utterances, response, embedding, smn))

    score = ComputeR10_1(predict, dev_examples[2])
    return score


max_length = 100
max_num_utterance = 6

n_epochs = 20
lr = 0.001
batch_size = 100
infer_batch_size = 4000
embed_size = 300
hidden_size = 200
compression_hidden_size = 50
bidirectional = False
dropout_p = 0.5
num_layers = 1
cut = 8

dev_id = 0
model_dir = 'smn-kb-model-' + str(dev_id)


def run_train():
    data = {
        'trainAnswers': [],
        'devAnswers': [],
        'trainQuestions': [],
        'devQuestions': []
    }
    data['trainAnswers'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-' + str(dev_id) + '/train.txt.tgt', encoding="utf-8").read().strip().split('\n')))
    data['trainQuestions'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-' + str(dev_id) + '/train.txt.src', encoding="utf-8").read().strip().split('\n')))
    data['devAnswers'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-' + str(dev_id) + '/val.txt.tgt', encoding="utf-8").read().strip().split('\n')))
    data['devQuestions'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-' + str(dev_id) + '/val.txt.src', encoding="utf-8").read().strip().split('\n')))

    # for debug
    '''
    data['trainAnswers'] = data['trainAnswers'][:5]
    data['trainQuestions'] = data['trainQuestions'][:5]
    data['devAnswers'] = [list(x) for x in data['trainAnswers']]
    data['devQuestions'] = [list(x) for x in data['trainQuestions']]
    '''

    # 生成词表（word）
    if os.path.exists(model_dir + '/vocab_word'):
        t = open(model_dir + '/vocab_word', 'rb')
        vocab_word = pickle.load(t)
        t.close()
    else:
        vocab_word = prepare_vocabulary(data, cut=cut)
        t = open(model_dir + '/vocab_word', 'wb')
        pickle.dump(vocab_word, t)
        t.close()
    print("========================word===========================")
    print('dec_vocab_size: ', vocab_word.n_words_for_decoder)
    print('vocab_size: ', vocab_word.n_words)
    print('max_word_length: ', max(map(lambda x: len(x), vocab_word.word2index)))

    # 生成数据（截断，生成负例）
    if os.path.exists(model_dir + '/data'):
        t = open(model_dir + '/data', 'rb')
        train_examples, dev_examples = pickle.load(t)
        t.close()
    else:
        train_examples = gen_data(vocab_word, data['trainQuestions'], data['trainAnswers'], 1, max_length, max_num_utterance)
        dev_examples = gen_data(vocab_word, data['devQuestions'], data['devAnswers'], 10, max_length, max_num_utterance)
        t = open(model_dir + '/data', 'wb')
        pickle.dump((train_examples, dev_examples), t)
        t.close()
    print(train_examples[0][1])
    print(train_examples[0][2])
    print(dev_examples[0][1])
    print(dev_examples[0][2])
    print("========================dataset===========================")
    print('train: ', len(train_examples[0]), len(train_examples[1]), len(train_examples[2]))
    print('dev: ', len(dev_examples[0]), len(dev_examples[1]), len(dev_examples[2]))

    embed = init_embedding(embed_size, vocab_word.n_words, vocab_word.word2index)
    embedding = nn.Embedding(vocab_word.n_words, embed_size, padding_idx=0).from_pretrained(embed, freeze=False)
    smn = SMN(embed_size, hidden_size, compression_hidden_size, max_num_utterance, max_length,
              bidirectional=bidirectional, num_layers=num_layers, dropout_p=dropout_p)

    embedding = torch.nn.DataParallel(embedding).to(device_cuda)
    smn = torch.nn.DataParallel(smn).to(device_cuda)

    if os.path.isfile(model_dir + '/embedding'):
        embedding.load_state_dict(torch.load(model_dir + '/embedding'))

    if os.path.isfile(model_dir + '/smn'):
        smn.load_state_dict(torch.load(model_dir + '/smn'))

    optimizer = optim.Adam([{"params": embedding.parameters()}, {"params": smn.parameters()}], lr=lr, amsgrad=True)

    if os.path.isfile(model_dir + '/optimizer'):
        optimizer.load_state_dict(torch.load(model_dir + '/optimizer'))

    trainIters(vocab_word, embedding, smn, optimizer, train_examples, dev_examples, n_epochs, lr, batch_size,
               infer_batch_size, print_every=1)


def prediction(input_data, candidate_answers):

    # 生成词表（word）
    t = open(model_dir + '/vocab_word', 'rb')
    vocab_word = pickle.load(t)
    t.close()
    print("========================word===========================")
    print('dec_vocab_size: ', vocab_word.n_words_for_decoder)
    print('vocab_size: ', vocab_word.n_words)
    print('max_word_length: ', max(map(lambda x: len(x), vocab_word.word2index)))

    # smn
    embedding = nn.Embedding(vocab_word.n_words, embed_size, padding_idx=0)
    smn = SMN(embed_size, hidden_size, compression_hidden_size, max_num_utterance, max_length,
              bidirectional=bidirectional, num_layers=num_layers, dropout_p=dropout_p)

    embedding = torch.nn.DataParallel(embedding).to(device_cuda)
    smn = torch.nn.DataParallel(smn).to(device_cuda)

    embedding.load_state_dict(torch.load(model_dir + '/embedding', map_location='cpu'))
    smn.load_state_dict(torch.load(model_dir + '/smn', map_location='cpu'))

    input_data_with_kb = list(map(lambda x: sentence_with_kb(x), input_data))
    input_data_with_kb = list(map(lambda x: word_tokenizer(x), input_data_with_kb))

    pred_probs = []
    for i in range(len(input_data)):
        # score
        candidate_response = candidate_answers[i]
        candidate_utterances = [input_data_with_kb[i]] * len(candidate_response)
        examples = gen_data(vocab_word, candidate_utterances, candidate_response, 1, max_length, max_num_utterance)
        print(candidate_utterances)
        print(candidate_response)
        print(examples[0][1])
        print(examples[0][2])
        utterances = torch.tensor(examples[0], dtype=torch.long, device=device_cuda)
        response = torch.tensor(examples[1], dtype=torch.long, device=device_cuda)
        probs = inference(utterances, response, embedding, smn)
        pred_probs.append(probs)

    return pred_probs


if __name__ == "__main__":
    run_train()
