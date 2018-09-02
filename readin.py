# -*- coding:utf-8 -*-

import io
import json
from csv import DictWriter
from functional import seq
import os
import pandas as pd
import numpy as np


def json2csv():
    with io.open('data.json', 'r', encoding='utf-8') as fr:
        data = []
        for line in fr.readlines():
            data.append(json.loads(line))
    ls = []
    seq(data).filter(lambda item: item['area'] == u'北京').map(lambda item: ls.extend(item['data'])).list()
    ls.sort(key=lambda item: item[u'date'])
    with io.open('data.csv', 'w', newline='') as fw:
        fields = ['date', 'aqi', 'class', 'pm2.5', 'pm10', 'so2', 'co', 'no2', 'o3']
        writer = DictWriter(fw, fieldnames=fields)
        writer.writeheader()
        for items in ls:
            writer.writerow({'date': items['date'],
                             'aqi': items['aqi'],
                             'class': items['class'],
                             'pm2.5': items['pm2.5'],
                             'pm10': items['pm10'],
                             'so2': items['so2'],
                             'co': items['co'],
                             'no2': items['no2'],
                             'o3': items['o3']})


def cut():
    path = "E:/Pycharm/Mywork/aqi/data/1.raw/WIN"
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s1, s2, s3 = [], [], []
    for file in files:
        with open(path + "/" + file, 'r') as fr:
            for line in fr.readlines():
                if line.split()[0] == '54406':
                    s1.append([int(i) for i in line.split()])
                elif line.split()[0] == '54416':
                    s2.append([int(i) for i in line.split()])
                elif line.split()[0] == '54511':
                    s3.append([int(i) for i in line.split()])
                else:
                    continue
    df1 = pd.DataFrame(data=np.array(s1))
    df2 = pd.DataFrame(data=np.array(s2))
    df3 = pd.DataFrame(data=np.array(s3))
    paths = 'E:/Pycharm/Mywork/aqi/data/2.preprocessed/meteorology/WIN'
    df1.to_excel(paths + '1.xlsx', sheet_name='54406', header=False, index=False)
    df2.to_excel(paths + '2.xlsx', sheet_name='54416', header=False, index=False)
    df3.to_excel(paths + '3.xlsx', sheet_name='54511', header=False, index=False)
    print(df1.shape, df2.shape, df3.shape)
    print('job done!')


if __name__ == "__main__":
    # json2csv()
    cut()
