# -*- coding: utf-8 -*-

import codecs
import logging
import os

from sklearn.model_selection import KFold


def data_processing():
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
    with codecs.open("data/chat.txt", mode = 'r', encoding = "utf-8") as rf:
        with codecs.open("data/questions.txt", mode="a", encoding="utf-8") as wf_question:
            with codecs.open("data/answers.txt", mode="a", encoding="utf-8") as wf_answer:
                try:
                    line = rf.readline()
                    while line:
                        splitline = line.strip('\r\n').split("\t")

                        # 直接跳过不存在的条目
                        if splitline[6].strip() == '':
                            line = rf.readline()
                            continue

                        if sessionId == splitline[0]:
                            try:
                                if splitline[2] == '0':
                                    # 生成对话
                                    if QAQAQ != '' and answer != '':
                                        wf_question.write(QAQAQ + "\n")
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
                                wf_question.write(QAQAQ + "\n")
                                wf_answer.write(answer + "\n")
                            sessionId = splitline[0]
                            question = ''
                            answer = ''
                            QAQAQ = ''
                            countQuestion = 0
                            countAnswer = 0
                            continue

                        line = rf.readline()

                    # 生成对话
                    if QAQAQ != '' and answer != '':
                        wf_question.write(QAQAQ + "\n")
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
 