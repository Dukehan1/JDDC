# -*- coding: utf-8 -*-

import re
from lxml import etree
import requests
import time

_DIGIT_RE = re.compile("\d+")
s = requests.session()
s.keep_alive = False

def get_info(sku_id):
    text = '#'
    while text == '#':
        try:
            text = requests.get('https://item.jd.com/' + sku_id + '.html').text
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue
    if text:
        html = etree.HTML(text)
        result = ''.join(html.xpath('//div[@class="sku-name"]/text()')).strip()
        result = re.sub(_DIGIT_RE, '[数字x]', result)
        result = result.replace('\r\n', '')
    else:
        result = ''
    print(sku_id, result)
    return result

def ware_processing():
    ware_file = open("preliminaryData/ware.txt", encoding="utf-8").read().strip().split('\n')[1:]
    order_file = open("preliminaryData/order.txt", encoding="utf-8").read().strip().split('\n')[1:]
    ware = {}
    for w in ware_file:
        w = w.split("\t")
        ware[w[0]] = w[1]
    for o in order_file:
        o = o.split("\t")
        ware[o[2]] = o[3]

    pr = open("preliminaryData/ware_o.txt", 'w', encoding="utf-8")
    pr.write('SKU\t品类\n')
    for k, v in ware.items():
        pr.write('\t'.join([k, v]) + '\n')

    pr = open("preliminaryData/ware_p.txt", 'w', encoding="utf-8")
    pr.write('SKU\t品类\n')
    for k, v in ware.items():
        pr.write('|||'.join([k, v, get_info(k)]) + '\n')


def ware_processing_r():
    ware_o_file = open("preliminaryData/ware_o.txt", encoding="utf-8").read().strip().split('\n')[1:]
    ware_p_file = open("preliminaryData/ware_p.txt", encoding="utf-8").read().strip().split('\n')[1:]
    ware_o = {}
    ware_p = {}
    for w in ware_o_file:
        w = w.split("\t")
        ware_o[w[0]] = w[1]
    for w in ware_p_file:
        w = w.split("|||")
        ware_p[w[0]] = (w[1], w[2])

    pr = open("preliminaryData/ware_p.txt", 'w', encoding="utf-8")
    pr.write('SKU\t品类\n')
    for k, v in ware_o.items():
        if k in ware_p:
            pr.write('|||'.join([k, v, ware_p[k][1]]) + '\n')
        else:
            pr.write('|||'.join([k, v, get_info(k)]) + '\n')


if __name__ == "__main__": 
    ware_processing_r()
 