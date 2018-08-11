# -*- coding: utf-8 -*-

import codecs
import logging
import re

from sklearn.model_selection import KFold

_ORDER_RE = re.compile("\[(ORDERID_\d+)\]")
_USER_RE = re.compile("\[(USERID_\d+)\]")
_URL_RE = re.compile("https?://item.jd.com/(\d+|\[数字x\]).html")

def data_processing():

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

    with codecs.open("data/questions.txt", mode="w", encoding="utf-8") as wfquestion:
        with codecs.open("data/answers.txt", mode="w", encoding="utf-8") as wfanswer:
            try:
                wfquestion.truncate()
                wfanswer.truncate()
            except Exception as e:
                logging.info("data_processing:clear data_processing.txt error:" + str(e))
            finally:
                wfquestion.close()
                wfanswer.close()
    
    question = ''
    answer = ''
    QAQAQ = ''
    countQuestion = 0
    countAnswer = 0
    sessionId = ''
    order_1 = '<ORDER_1_N>'
    user_1 = '<USER_1_N>'
    user_2 = '<USER_2_N>'
    sku_1 = ''
    sku_2 = ''

    with codecs.open("preliminaryData/chat.txt", mode = 'r', encoding = "utf-8") as rf:
        with codecs.open("data/questions.txt", mode="a", encoding="utf-8") as wf_question:
            with codecs.open("data/answers.txt", mode="a", encoding="utf-8") as wf_answer:
                try:
                    lines = rf.readlines()
                    i = 0

                    while i < len(lines):
                        splitline = lines[i].strip('\r\n').split("\t")

                        # 直接跳过不存在的条目
                        if splitline[6].strip() == '':
                            i += 1
                            continue

                        if sessionId == splitline[0]:
                            try:
                                if splitline[2] == '0':
                                    # 生成对话
                                    if QAQAQ != '' and answer != '':
                                        wf_question.write(
                                            QAQAQ + '<s>'.join([order_1, user_1, user_2, sku_1, sku_2]) + '<s>' + "\n")
                                        wf_answer.write(answer + "\n")

                                    if answer != '':
                                        # 如果一段对话以A起始的话则忽略第一条A
                                        if QAQAQ != '':
                                            QAQAQ = QAQAQ + answer + '<s>'
                                        answer = ''
                                        countAnswer = countAnswer + 1

                                    if question == '':
                                        question = question + splitline[6].strip()
                                    else:
                                        question = question + u'，' + splitline[6].strip()

                                elif splitline[2] == '1':
                                    if question != '':
                                        QAQAQ = QAQAQ + question + '<s>'
                                        question = ''
                                        countQuestion = countQuestion + 1

                                    if answer == '':
                                        answer = answer + splitline[6].strip()
                                    else:
                                        answer = answer + u'，' + splitline[6].strip()

                            except Exception as e:
                                logging.error("data_processing:write into chatmasked_user failure" + str(e))
                        else:
                            # 生成对话
                            if QAQAQ != '' and answer != '':
                                wf_question.write(
                                    QAQAQ + '<s>'.join([order_1, user_1, user_2, sku_1, sku_2]) + '<s>' + "\n")
                                wf_answer.write(answer + "\n")
                            sessionId = splitline[0]
                            question = ''
                            answer = ''
                            QAQAQ = ''
                            countQuestion = 0
                            countAnswer = 0

                            # 更新knowledge
                            order_1 = '<ORDER_1_N>'
                            user_1 = '<USER_1_N>'
                            user_2 = '<USER_2_N>'
                            sku_1 = ''
                            sku_2 = ''
                            j = i
                            while j < len(lines) and sessionId == lines[j].strip('\r\n').split("\t")[0]:
                                temp = lines[j].strip('\r\n').split("\t")
                                user_id = temp[1]
                                if user_id in user:
                                    user_1 = '<USER_1_' + user[user_id][1] + '>'
                                    user_2 = '<USER_2_' + user[user_id][3] + '>'
                                if re.search(_URL_RE, temp[6]):
                                    sku_id = re.search(_URL_RE, temp[6]).group(1)
                                    if sku_id in sku:
                                        sku_1 = sku[sku_id][1]
                                        sku_2 = sku[sku_id][2]
                                if re.search(_ORDER_RE, temp[6]):
                                    order_id = re.search(_ORDER_RE, temp[6]).group(1)
                                    if order_id in order:
                                        sku_1s = []
                                        sku_2s = []
                                        for term in order[order_id]:
                                            order_1 = '<ORDER_1_' + term[4] + '>'
                                            sku_1s.append(sku[term[2]][1])
                                            sku_2s.append(sku[term[2]][2])
                                        sku_1 = '|'.join(sku_1s)
                                        sku_2 = '|'.join(sku_2s)
                                j += 1
                            continue

                        i += 1

                    # 生成对话
                    if QAQAQ != '' and answer != '':
                        wf_question.write(
                            QAQAQ + '<s>'.join([order_1, user_1, user_2, sku_1, sku_2]) + '<s>' + "\n")
                        wf_answer.write(answer + "\n")

                except Exception as e:
                    logging.error("data_processing: data processing failure!" + str(e))
                finally:
                    rf.close()
                    wf_question.close()
                    wf_answer.close()
   
def cutData():

    X = open('data/questions.txt', encoding="utf-8").read().strip().split('\n')
    y = open('data/answers.txt', encoding="utf-8").read().strip().split('\n')

    skf = KFold(n_splits=10, random_state=2333, shuffle=True)
    i = 0
    for train_index, dev_index in skf.split(X, y):
        print(dev_index)
        p1 = open('data/questions-' + str(i) + '.txt', 'w', encoding='utf-8')
        p1.write('\n'.join([X[i] for i in dev_index]))
        p1.close()
        p2 = open('data/answers-' + str(i) + '.txt', 'w', encoding='utf-8')
        p2.write('\n'.join([y[i] for i in dev_index]))
        p2.close()
        i += 1
                            
    # os.remove("questions.txt")
    # os.remove("answers.txt")
                            
                
if __name__ == "__main__": 
    data_processing()
    cutData()
 