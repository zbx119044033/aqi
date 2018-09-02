# -*- coding:utf-8 -*-

import io
import json
from csv import DictWriter
from functional import seq


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

if __name__ == "__main__":
    json2csv()
