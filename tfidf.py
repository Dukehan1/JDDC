#-*-coding:utf-8 -*-
import codecs
import time
import jieba
import os
import re
import logging
from gensim import corpora, models, similarities
from collections import defaultdict
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction

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


def remove_stopwords_and_slice(seg_list, stopwords):
    # slice QAQAQ + <s>order_1<s>user_1<s>user_2<s>sku_1<s>sku_2<s> and remove stopwords
    results = []
    s_count = 0
    for seg in reversed(seg_list):
        if seg == '<s>':
            s_count += 1
            if s_count == 11:
                break
            continue

        if seg in stopwords or s_count < 6:
            continue
        results.append(seg)

    return list(reversed(results))


def custom_print(print_string):
    localtime = time.asctime(time.localtime(time.time()))
    print(print_string + ' ' + localtime)


def run_train(min_frequency=1):
    custom_print("===================prepare data===================")
    trainQuestions = list(map(lambda x: x.split(' '),
                              open('opennmt-kb/train.txt.src', encoding="utf-8").read().strip().split('\n'))) + \
                     list(map(lambda x: x.split(' '),
                              open('opennmt-kb/val.txt.src', encoding="utf-8").read().strip().split('\n')))

    results = []
    stopwords = load_stopwords()
    for sentence in trainQuestions:
        results.append(remove_stopwords_and_slice(sentence, stopwords))
    texts = results
    custom_print("===================train model===================")
    # 构建其他复杂模型前需要的简单模型
    custom_print("=================== create simple model==================")
    ## 删除低频词
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > min_frequency] for text in texts]

    dictionary = corpora.Dictionary(texts)
    if not os.path.exists('model_tfidf'):
        os.makedirs('model_tfidf')
    dictionary.save('model_tfidf/dictionary.dict')
    corpus_simple = [dictionary.doc2bow(text) for text in texts]

    # 转换模型
    custom_print("===================transfer model==================")

    model = models.TfidfModel(corpus_simple)
    model.save('model_tfidf/tfidf.model')
    corpus = model[corpus_simple]

    # 创建相似度矩阵
    custom_print("===================create matrix==================")
    if not os.path.exists('model_tfidf/index'):
        os.makedirs('model_tfidf/index')
    # index = similarities.MatrixSimilarity(corpus)
    index = similarities.Similarity('model_tfidf/index/index', corpus, num_features=len(dictionary))
    index.save('model_tfidf/similarity.index')


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
    # 将' '全部替换成''，否则文件无法处理
    tokens = [word.strip() for word in tokens if word]
    return tokens


def run_prediction(input_file_path, output_file_path):
    trainAnswers = list(map(lambda x: x.split(' '),
                            open('opennmt-kb/train.txt.tgt', encoding="utf-8").read().strip().split('\n'))) + \
                   list(map(lambda x: x.split(' '),
                            open('opennmt-kb/val.txt.tgt', encoding="utf-8").read().strip().split('\n')))

    dictionary = corpora.Dictionary.load('model_tfidf/dictionary.dict')
    model = models.TfidfModel.load('model_tfidf/tfidf.model')
    index = similarities.Similarity.load('model_tfidf/similarity.index')

    input_data = open(input_file_path, encoding="utf-8").read().strip().split('\n')
    input_data = list(map(lambda x: word_tokenizer(x), input_data))

    stopwords = load_stopwords()
    pred = []
    for i in range(len(input_data)):
        input_query = input_data[i]
        result = []
        for word in input_query:
            if word in stopwords:
                continue
            result.append(word)
        input_query = result
        vec_bow = dictionary.doc2bow(input_query)
        sentence_vec = model[vec_bow]
        sims = index[sentence_vec]
        answer_index = max(list(enumerate(sims)), key=lambda item: item[1])[0]
        pred_query = trainAnswers[answer_index]
        pred_query = [word if word != '' else ' ' for word in pred_query]
        pred.append(''.join(pred_query))
    output_file = open(output_file_path, 'w', encoding='utf-8')
    output_file.write('\n'.join(pred))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    run_prediction('questions50.txt', 'result.txt')
