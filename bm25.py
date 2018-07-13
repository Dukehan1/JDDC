# -*- coding: utf-8 -*-

import jieba
from gensim import corpora
from gensim.summarization import bm25
import re

K = 10

# Regular expressions used to tokenize.
_DIGIT_RE = re.compile("\d+")


def word_tokenizer(sentence):
    # word level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    candidate = sentence.split('<s>')
    result = []
    for c in candidate:
        if c:
            tokens = list(jieba.cut(c))
            tokens = [word.encode('utf-8') for word in tokens if word]
            result.extend(tokens)
            result.append('<s>')
    return result


def char_tokenizer(sentence):
    # char level
    sentence = re.sub(_DIGIT_RE, "0", sentence)
    candidate = sentence.split('<s>')
    result = []
    for c in candidate:
        if c:
            tokens = list(c.decode('utf-8'))
            tokens = [word.encode('utf-8') for word in tokens if word]
            result.extend(tokens)
            result.append('<s>')
    return result

corpus = []
lines = open('data/questions.txt').read().strip().split('\n')
for line in lines[:20]:
    tokens = word_tokenizer(line)
    corpus.append(tokens)

dictionary = corpora.Dictionary(corpus)
# print len(corpus)
# print len(dictionary)

bm25Model = bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

query = word_tokenizer('咨询订单号:[ORDERID_10005629] 订单金额:[金额x] 下单时间:[日期x]<s>')
print query
scores = bm25Model.get_scores(query, average_idf)
print scores

idx = scores.index(max(scores))
print idx