# -*- coding:utf-8 -*-

import json
import io
from functional import seq
import csv
import pickle
import numpy as np


def clean():
    with open("gold_product4.json", "r") as fr:
        content = json.load(fr)

        def process_kv(kvs):
            res = []
            for k, v in kvs.items():
                res.append({"name": k, "value": v})
            return res

        def process_prop(prop):
            if prop["name"] == u"保障利益":
                prop["value"] = seq(prop["value"]).map(process_kv).list()
            return prop

        def process_item(item):
            item["domain"] = "product"
            del item["_id"]
            item["props"] = seq(item["props"]).map(process_prop).list()
            return item

        result = seq(content).map(process_item).list()
        return result


def explore():
    with open('pollute.json', 'r') as fr:
        data_list = []
        for line in fr.readlines():
            data_list.append(json.loads(line))
    data_bj, data_lf, data_zjk, data_cd, data_ts, data_tj = [], [], [], [], [], []
    for item in data_list:
        if item['area'] == u'北京':
            for d in item['data']:
                data_bj.append(d['date'])
        elif item['area'] == u'廊坊':
            for d in item['data']:
                data_lf.append(d['date'])
        elif item['area'] == u'张家':
            for d in item['data']:
                data_zjk.append(d['date'])
        elif item['area'] == u'承德':
            for d in item['data']:
                data_cd.append(d['date'])
        elif item['area'] == u'唐山':
            for d in item['data']:
                data_ts.append(d['date'])
        elif item['area'] == u'天津':
            for d in item['data']:
                data_tj.append(d['date'])
    data_bj.sort()
    data_lf.sort()
    data_zjk.sort()
    data_cd.sort()
    data_ts.sort()
    data_tj.sort()
    print('北京: %d' % len(data_bj))
    print(data_bj)
    print('廊坊: %d' % len(data_lf))
    print(data_lf)
    print('张家口: %d' % len(data_zjk))
    print(data_zjk)
    print('承德: %d' % len(data_cd))
    print(data_cd)
    print('唐山: %d' % len(data_ts))
    print(data_ts)
    print('天津: %d' % len(data_tj))
    print(data_tj)


def divide():
    with open('pollute.json', 'r') as fr:
        data_list = []
        for line in fr.readlines():
            data_list.append(json.loads(line))
    data_bj, data_lf, data_zjk, data_cd, data_ts, data_tj = [], [], [], [], [], []
    for item in data_list:
        if item['area'] == u'北京':
            data_bj.extend(item['data'])
        elif item['area'] == u'廊坊':
            data_lf.extend(item['data'])
        elif item['area'] == u'张家':
            data_zjk.extend(item['data'])
        elif item['area'] == u'承德':
            data_cd.extend(item['data'])
        elif item['area'] == u'唐山':
            data_ts.extend(item['data'])
        elif item['area'] == u'天津':
            data_tj.extend(item['data'])
    data_bj.sort(key=lambda d: int(d['date']))
    data_lf.sort(key=lambda d: int(d['date']))
    data_zjk.sort(key=lambda d: int(d['date']))
    data_cd.sort(key=lambda d: int(d['date']))
    data_ts.sort(key=lambda d: int(d['date']))
    data_tj.sort(key=lambda d: int(d['date']))
    print(len(data_bj), len(data_lf), len(data_zjk), len(data_cd), len(data_ts), len(data_tj))
    data_bj.extend(data_lf)
    data_bj.extend(data_zjk)
    data_bj.extend(data_cd)
    data_bj.extend(data_ts)
    data_bj.extend(data_tj)
    print(len(data_bj))
    with open('pollutants.json', 'w') as fw:
        for item in data_bj:
            line = json.dumps(dict(item), ensure_ascii=False) + "\n"
            fw.write(line)
    print('job done')


def json2csv():
    with open('beijing.csv', 'wb') as csv_file:
        fieldnames = ["date", "aqi", "pm2.5", "pm10", "co", "no2", "so2", "o3", "class"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        with open('temp.json', 'r') as data_file:
            for line in data_file.readlines():
                data = json.loads(line, encoding='utf-8')
                print(data['date'])
                writer.writerow({"date": data['date'],
                                 "aqi": data['aqi'],
                                 "pm2.5": data['pm2.5'],
                                 "pm10": data['pm10'],
                                 "co": data['co'],
                                 "no2": data['no2'],
                                 "so2": data['so2'],
                                 "o3": data['o3'],
                                 "class": data['class']})
    print("job done")


def my_dump():
    with open('E:/Pycharm/mywork/aqi/data/zhangjiakou.csv', 'rb') as fr:
        data_mat = []
        data_reader = csv.reader(fr)
        i = 0
        for row in data_reader:
            if i == 0:
                i += 1
                continue
            else:
                sample = np.array([int(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8])])
                data_mat.append(sample)
                i += 1
    print('read finished', len(data_mat))

    with open('E:/Pycharm/mywork/aqi/data/zhangjiakou.pkl', 'wb') as fw:
        pickle.dump(np.array(data_mat), fw)
    print('dump finished')


def dump_data():
    with io.open('E:/Pycharm/Mywork/aqi/data/raw/AQI/all.csv', 'r', encoding='utf-8') as fr:
        data_mat = []
        data_reader = csv.reader(fr)
        for row in data_reader:
            print(type(row))
            sample = np.array([float(row[i]) for i in range(145)])
            data_mat.append(sample)
    print('read finished', len(data_mat))

    with open('E:/Pycharm/Mywork/aqi/data/raw/AQI/all.pkl', 'wb') as fw:
        pickle.dump(np.array(data_mat), fw)
    print('dump finished')


if __name__ == "__main__":
    # explore()
    # divide()
    # json2csv()
    # my_dump()
    dump_data()
