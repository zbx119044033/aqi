# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from functional import seq
from scipy.special import boxcox1p


category = ['year', 'month', 'class-aqi', 'class-bj-1', 'class-lf-1', 'class-ts-1', 'class-tj-1', 'class-zjk-1',
            'class-cd-1', 'SimpClass-cd-1', 'SimpClass-lf-1', 'SimpClass-tj-1', 'SimpClass-ts-1', 'SimpClass-zjk-1',
            'class-bj-1-s2', 'class-bj-1-s3', 'class-bj-1-sq']


def drop(frame):
    # 舍去离群点( abs(aqi - ap1) >300: 6; 	>290: 7; 	>280:8; 	>270: 11; 	>260: 12; 	>250: 13)
    aqi = frame['obj'].values
    ap1 = frame['aqi-bj-1'].values
    drop_id = []
    for i in range(len(aqi)):
        if abs(aqi[i] - ap1[i]) > 250:
            drop_id.append(i)
    dropped = frame.drop(drop_id)
    print('dropped %d items' % len(drop_id))
    return dropped


def extract(frame):
    # 对top10特征进行二次提取
    # 1* Simplifications of existing features
    frame["SimpClass-cd-1"] = frame['class-cd-1'].replace({1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4})
    frame["SimpClass-lf-1"] = frame['class-lf-1'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5})
    frame["SimpClass-tj-1"] = frame['class-tj-1'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4})
    frame["SimpClass-ts-1"] = frame['class-ts-1'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4})
    frame["SimpClass-zjk-1"] = frame['class-zjk-1'].replace({1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3})

    # 2* Combinations of existing features
    frame["OvallGrd-bj-1"] = frame["class-bj-1"] * frame["aqi-bj-1"]
    frame["pm25Scr-bj-1"] = frame["aqi-bj-1"] * frame["pm2.5-bj-1"]
    frame["pm25Grd-bj-1"] = frame["class-bj-1"] * frame["pm2.5-bj-1"]
    frame["pm10Scr-bj-1"] = frame["aqi-bj-1"] * frame["pm10-bj-1"]
    frame["pm10Grd-bj-1"] = frame["class-bj-1"] * frame["pm10-bj-1"]
    frame["no2Scr-bj-1"] = frame["aqi-bj-1"] * frame["no2-bj-1"]
    frame["no2Grd-bj-1"] = frame["class-bj-1"] * frame["no2-bj-1"]
    frame["pollute-bj-1"] = frame["pm2.5-bj-1"] + 0.8 * frame["pm10-bj-1"] + 0.5 * frame["no2-bj-1"]
    # 廊坊
    frame["OvallGrd-lf-1"] = frame["class-lf-1"] * frame["aqi-lf-1"]
    frame["pm25Scr-lf-1"] = frame["aqi-lf-1"] * frame["pm2.5-lf-1"]
    frame["pm25Grd-lf-1"] = frame["class-lf-1"] * frame["pm2.5-lf-1"]
    frame["pm10Scr-lf-1"] = frame["aqi-lf-1"] * frame["pm10-lf-1"]
    frame["pm10Grd-lf-1"] = frame["class-lf-1"] * frame["pm10-lf-1"]
    frame["pollute-lf-1"] = frame["pm2.5-lf-1"] + 0.8 * frame["pm10-lf-1"]

    # 3* Polynomials on the top 10 existing features
    # aqi-bj-1
    frame["aqi-bj-1-s2"] = frame["aqi-bj-1"] ** 2
    frame["aqi-bj-1-s3"] = frame["aqi-bj-1"] ** 3
    frame["aqi-bj-1-sq"] = np.sqrt(frame["aqi-bj-1"])
    # no2Grd-bj-1
    frame["no2Grd-bj-1-s2"] = frame["no2Grd-bj-1"] ** 2
    frame["no2Grd-bj-1-s3"] = frame["no2Grd-bj-1"] ** 3
    frame["no2Grd-bj-1-sq"] = np.sqrt(frame["no2Grd-bj-1"])
    # OvallGrd-bj-1
    frame["OvallGrd-bj-1-s2"] = frame["OvallGrd-bj-1"] ** 2
    frame["OvallGrd-bj-1-s3"] = frame["OvallGrd-bj-1"] ** 3
    frame["OvallGrd-bj-1-sq"] = np.sqrt(frame["OvallGrd-bj-1"])
    # pollute-bj-1
    frame["pollute-bj-1-s2"] = frame["pollute-bj-1"] ** 2
    frame["pollute-bj-1-s3"] = frame["pollute-bj-1"] ** 3
    frame["pollute-bj-1-sq"] = np.sqrt(frame["pollute-bj-1"])
    # no2Scr-bj-1
    frame["no2Scr-bj-1-s2"] = frame["no2Scr-bj-1"] ** 2
    frame["no2Scr-bj-1-s3"] = frame["no2Scr-bj-1"] ** 3
    frame["no2Scr-bj-1-sq"] = np.sqrt(frame["no2Scr-bj-1"])
    # pm2.5-bj-1
    frame["pm2.5-bj-1-s2"] = frame["pm2.5-bj-1"] ** 2
    frame["pm2.5-bj-1-s3"] = frame["pm2.5-bj-1"] ** 3
    frame["pm2.5-bj-1-sq"] = np.sqrt(frame["pm2.5-bj-1"])
    # pm25Grd-bj-1
    frame["pm25Grd-bj-1-s2"] = frame["pm25Grd-bj-1"] ** 2
    frame["pm25Grd-bj-1-s3"] = frame["pm25Grd-bj-1"] ** 3
    frame["pm25Grd-bj-1-sq"] = np.sqrt(frame["pm25Grd-bj-1"])
    # class-bj-1
    frame["class-bj-1-s2"] = frame["class-bj-1"] ** 2
    frame["class-bj-1-s3"] = frame["class-bj-1"] ** 3
    frame["class-bj-1-sq"] = np.sqrt(frame["class-bj-1"])
    # pm10Grd-bj-1
    frame["pm10Grd-bj-1-s2"] = frame["pm10Grd-bj-1"] ** 2
    frame["pm10Grd-bj-1-s3"] = frame["pm10Grd-bj-1"] ** 3
    frame["pm10Grd-bj-1-sq"] = np.sqrt(frame["pm10Grd-bj-1"])
    # pm10-bj-1
    frame["pm10-bj-1-s2"] = frame["pm10-bj-1"] ** 2
    frame["pm10-bj-1-s3"] = frame["pm10-bj-1"] ** 3
    frame["pm10-bj-1-sq"] = np.sqrt(frame["pm10-bj-1"])
    # calculate pearson coefficients
    # corr = frame.corr()
    # corr.sort_values(["obj"], ascending=False, inplace=True)
    # print(corr.obj)
    # corr.to_csv('E:/Pycharm/Mywork/aqi/log/FE/pearson.csv')
    return frame


def transform(frame):
    # 对数值型偏度特征进行log转换(abs(skewness) > 0.8)
    numerical = seq(frame.columns).filter(lambda s: s not in category).list()
    train_num = frame.loc[:, numerical]
    skewness = train_num.apply(lambda x: stats.skew(x))
    skewness = skewness[abs(skewness) > 0.8]
    skewed_features = skewness.index
    # Box Cox Transformation of (highly) skewed features
    train_num[skewed_features] = np.log1p(train_num[skewed_features])
    # train_num[skewed_features] = boxcox1p(train_num[skewed_features], 0.15)
    print("log transformed %d skewed numerical features" % skewness.shape[0])
    return train_num


def dummy(frame):
    # 对category变量进行哑铃编码
    train_cat = frame.loc[:, category]
    train_cat = pd.get_dummies(train_cat, columns=category)
    print("dummy encoded %d categorical features" % len(category))
    return train_cat


def dump(frame):
    dump_path = 'E:/Pycharm/Mywork/aqi/data/feature/'
    frame.to_csv(dump_path+'featured-2.csv')
    frame.to_pickle(dump_path+'featured-2.pkl')
    print('dumped !')


if __name__ == "__main__":
    path = 'E:/Pycharm/Mywork/aqi/data/offshelf/aqi-all-1.csv'
    data = pd.read_csv(path)
    data1 = drop(data)
    features = extract(data1)
    num = transform(features)
    cat = dummy(features)
    df = pd.concat([num, cat], axis=1)
    # print(df.shape)
    # dump(df)
