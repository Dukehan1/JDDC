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
    sentence_with_kb, char_tokenizer


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
        output_prob = F.softmax(self.final(v.view(b, -1)), dim=1)  # (b, 2)
        return output, output_prob


def train(utterances, response, label, embedding, dam, optimizer, criterion):
    # Training mode (enable dropout)
    dam.train()
    optimizer.zero_grad()

    utterances = embedding(utterances)
    response = embedding(response)
    output, _ = dam(utterances, response)
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
        _, output = dam(utterances, response)
        output = output[:, 1]
        output = output.data.cpu().numpy()

        return output


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
        torch.save(dam.state_dict(), model_dir + '/dam')
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
            torch.save(dam.state_dict(), model_dir + '/dam')
            torch.save(embedding.state_dict(), model_dir + '/embedding')
            torch.save(optimizer.state_dict(), model_dir + '/optimizer')
            print('new ' + model_dir + ' saved.')
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

dev_id = 0
model_dir = 'dam-char-kb-model-' + str(dev_id)


def run_train():
    data = {
        'trainAnswers': [],
        'devAnswers': [],
        'trainQuestions': [],
        'devQuestions': []
    }
    data['trainAnswers'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-char-' + str(dev_id) + '/train.txt.tgt', encoding="utf-8").read().strip().split('\n')))
    data['trainQuestions'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-char-' + str(dev_id) + '/train.txt.src', encoding="utf-8").read().strip().split('\n')))
    data['devAnswers'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-char-' + str(dev_id) + '/val.txt.tgt', encoding="utf-8").read().strip().split('\n')))
    data['devQuestions'].extend(
        map(lambda x: x.split(' '), open('opennmt-kb-char-' + str(dev_id) + '/val.txt.src', encoding="utf-8").read().strip().split('\n')))

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

    embed = init_embedding(embed_size, vocab_word.n_words, vocab_word.word2index, True)
    embedding = nn.Embedding(vocab_word.n_words, embed_size, padding_idx=0).from_pretrained(embed, freeze=False)
    dam = DAM(embed_size, max_num_utterance, max_length, max_stacks)

    embedding = torch.nn.DataParallel(embedding).to(device_cuda)
    dam = torch.nn.DataParallel(dam).to(device_cuda)

    if os.path.isfile(model_dir + '/embedding'):
        embedding.load_state_dict(torch.load(model_dir + '/embedding'))

    if os.path.isfile(model_dir + '/dam'):
        dam.load_state_dict(torch.load(model_dir + '/dam'))

    optimizer = optim.Adam([{"params": embedding.parameters()}, {"params": dam.parameters()}], lr=lr, amsgrad=True)

    if os.path.isfile(model_dir + '/optimizer'):
        optimizer.load_state_dict(torch.load(model_dir + '/optimizer'))

    trainIters(vocab_word, embedding, dam, optimizer, train_examples, dev_examples, n_epochs, lr, batch_size,
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

    # dam
    embedding = nn.Embedding(vocab_word.n_words, embed_size, padding_idx=0)
    dam = DAM(embed_size, max_num_utterance, max_length, max_stacks)

    embedding = torch.nn.DataParallel(embedding).to(device_cuda)
    dam = torch.nn.DataParallel(dam).to(device_cuda)

    embedding.load_state_dict(torch.load(model_dir + '/embedding', map_location='cpu'))
    dam.load_state_dict(torch.load(model_dir + '/dam', map_location='cpu'))

    input_data_with_kb = list(map(lambda x: sentence_with_kb(x), input_data))
    input_data_with_kb = list(map(lambda x: char_tokenizer(x), input_data_with_kb))

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
        probs = inference(utterances, response, embedding, dam)
        pred_probs.append(probs)

    return pred_probs


if __name__ == "__main__":
    run_train()
