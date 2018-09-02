# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import boxcox1p
from sklearn.linear_model import RandomizedLasso
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def drop(frame):
    # 舍去离群点( abs(pm - pm1) >300: 5; 	>290: 6; 	>280:6; 	>270: 6; 	>260: 7; 	>250: 10)
    pm = frame['target'].values
    pm1 = frame['pm2.5-bj'].values
    drop_id = []
    for i in range(len(pm)):
        if abs(pm[i] - pm1[i]) > 250:
            drop_id.append(i)
    dropped = frame.drop(drop_id)
    print('dropped %d items' % len(drop_id))
    return dropped


def extract(frame):
    # 1* Simplifications of existing features
    frame["SimpClass-cd"] = frame['class-cd'].replace({1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}).astype('category')
    frame["SimpClass-lf"] = frame['class-lf'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}).astype('category')
    frame["SimpClass-tj"] = frame['class-tj'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}).astype('category')
    frame["SimpClass-ts"] = frame['class-ts'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}).astype('category')
    frame["SimpClass-zjk"] = frame['class-zjk'].replace({1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}).astype('category')

    # 2* Polynomials on the top 10 existing features
    # pm10-bj-1
    frame["pm10-bj-sqt"] = np.sqrt(frame["pm10-bj"])
    # co-bj-1
    frame["co-bj-sqt"] = np.sqrt(frame["co-bj"])
    # so2-bj-1
    frame["so2-bj-sqt"] = np.sqrt(frame["so2-bj"])
    # o3-bj-1
    frame["o3-bj-poly"] = frame["o3-bj"] + frame["o3-bj"] ** 2 + frame["o3-bj"] ** 3
    # pm2.5-bj-1
    frame["pm2.5-bj-sqt"] = np.sqrt(frame["pm2.5-bj"])
    # aqi-bj-1
    frame["aqi-bj-sqt"] = np.sqrt(frame["aqi-bj"])
    # so2-lf-1
    frame["so2-lf-sqt"] = np.sqrt(frame["so2-lf"])
    # pm2.5-lf-1
    frame["pm2.5-lf-sqt"] = np.sqrt(frame["pm2.5-lf"])
    # aqi-lf-1
    frame["aqi-lf-sqt"] = np.sqrt(frame["aqi-lf"])
    # co-lf-1
    frame["co-lf-sqt"] = np.sqrt(frame["co-cd"])
    # so2-tj-1
    frame["pm10-tj-poly"] = frame["pm10-tj"] + frame["pm10-tj"] ** 2
    # no2-cd-1
    frame["no2-cd-poly"] = frame["no2-cd"] + frame["no2-cd"] ** 2 + frame["no2-cd"] ** 3
    frame["no2-cd-s2"] = frame["no2-cd"] ** 2
    # so2-cd-1
    frame["so2-cd-poly"] = frame["so2-cd"] + frame["so2-cd"] ** 2
    # co-cd-1
    frame["co-cd-s2"] = frame["co-cd"] ** 2
    frame["co-cd-sqt"] = np.sqrt(frame["co-cd"])
    # no2-bd-1
    frame["no2-bd-poly"] = frame["no2-bd"] + frame["no2-bd"] ** 2 + frame["no2-bd"] ** 3
    # so2-ts-1
    frame['so2-ts-poly'] = frame['so2-ts'] + frame['so2-ts'] ** 2
    # no2-zjk-1
    frame["no2-zjk-s2"] = frame["no2-zjk"] ** 2

    # 3* nonlinear extraction
    frame["WD-max-sin"] = abs(np.sin((frame["WD-max"] - 1) * 1 / 8 * np.pi))
    frame["WD-max-cos"] = abs(np.cos((frame["WD-max"] - 1) * 1 / 8 * np.pi))
    frame["WD-max-sint"] = abs(np.sin((frame["WD-max"] - 1) * 1 / 8 * np.pi + 4 / 3 * np.pi))
    # frame["WD-max-sine"] = 54.15 + 51.454 * np.sin((frame["WD-max"] - 26.62) / 12.64 * np.pi)
    frame["WD-ext-sin"] = abs(np.sin((frame["WD-ext"] - 1) * 1 / 8 * np.pi))
    frame["WD-ext-cos"] = abs(np.cos((frame["WD-ext"] - 1) * 1 / 8 * np.pi))
    frame["WD-ext-sint"] = abs(np.sin((frame["WD-ext"] - 1) * 1 / 8 * np.pi + 4 / 3 * np.pi))
    # frame["WD-ext-sine"] = 30.63 - 72.41 * np.sin((frame["WD-ext"] + 16.88) / 16.556 * np.pi)
    # exponential fittings
    frame["o3-bj-exp"] = np.exp(- frame["o3-bj"])
    frame["o3-lf-exp"] = np.exp(- frame["o3-lf"])
    frame["so2-tj-exp"] = np.exp(- frame["so2-tj"])
    frame["o3-tj-exp"] = np.exp(- frame["o3-tj"])
    frame["o3-zjk-exp"] = np.exp(- frame["o3-zjk"])
    frame["o3-cd-exp"] = np.exp(- frame["o3-cd"])
    frame["o3-bd-exp"] = np.exp(- frame["o3-bd"])
    frame["o3-ts-exp"] = np.exp(- frame["o3-ts"])
    frame['EVP-exp'] = np.exp(-frame['EVP'])
    frame['WIN-mean-exp'] = np.exp(-frame['WIN-mean'])
    frame['WIN-max-exp'] = np.exp(-frame['WIN-max'])
    frame['WIN-ext-exp'] = np.exp(-frame['WIN-ext'])
    # logarithm fittings
    frame['pm2.5-zjk-log'] = np.log(frame['pm2.5-zjk'])
    frame['pm10-zjk-log'] = np.log(frame['pm10-zjk'])
    frame['so2-zjk-log'] = np.log(frame['so2-zjk'])
    frame['pm2.5-cd-log'] = np.log(frame['pm2.5-cd'])
    frame['pm10-cd-log'] = np.log(frame['pm10-cd'])
    frame['so2-bd-log'] = np.log(frame['so2-bd'])
    frame['co-bd-log'] = np.log(frame['co-bd'])

    # 4* Combinations of existing features
    frame['WIN-max-project1'] = frame['WIN-max-exp'] * frame["WD-max-sint"]
    # frame['WIN-max-project2'] = frame['WIN-max-exp'] * frame["WD-max-sine"]
    frame['WIN-ext-project1'] = frame['WIN-ext-exp'] * frame["WD-ext-sint"]
    # frame['WIN-ext-project2'] = frame['WIN-ext-exp'] * frame["WD-ext-sine"]
    frame["WIN-max-sin"] = frame["WIN-max"] * frame["WD-max-sin"]
    frame["WIN-max-cos"] = frame["WIN-max"] * frame["WD-max-cos"]
    frame["WIN-max-sint"] = frame["WIN-max"] * frame["WD-max-sint"]
    # frame["WIN-max-sine"] = frame["WIN-max"] * frame["WD-max-sine"]
    frame["WIN-ext-sin"] = frame["WIN-ext"] * frame["WD-ext-sin"]
    frame["WIN-ext-cos"] = frame["WIN-ext"] * frame["WD-ext-cos"]
    frame["WIN-ext-sint"] = frame["WIN-ext"] * frame["WD-ext-sint"]
    # frame["WIN-ext-sine"] = frame["WIN-ext"] * frame["WD-ext-sine"]
    print('extraction completed!')
    return frame


def transform(frame):
    # transfer category format to int64
    # cat_features = frame.dtypes[frame.dtypes == 'category'].index
    # frame[cat_features] = frame[cat_features].apply(pd.to_numeric)
    # log/Box-Cox transformation for (highly) skewed features(abs(skewness) > 0.8)
    numerical = frame.dtypes[frame.dtypes != 'category'].index
    # print(list(numerical))
    skewness = frame[numerical].apply(lambda x: stats.skew(x))
    skewed_features = skewness[abs(skewness) > 0.8].index
    frame[skewed_features] = np.log1p(frame[skewed_features])
    # train_num[skewed_features] = boxcox1p(train_num[skewed_features], 0.15)
    print("log transformed %d skewed numerical features" % skewness.shape[0])
    # dummy encode for category features
    print("dummy encoded %d categorical features" % len(frame.dtypes[frame.dtypes == 'category'].index))
    frame = pd.get_dummies(frame)
    return frame


def stability(frame):
    data_y = frame['target']
    data_x = frame.drop('target', axis=1)
    selection = RandomizedLasso(alpha=0.0011, scaling=0.8, sample_fraction=0.6, max_iter=100000).fit_transform(data_x, data_y)
    print(selection.shape)


def pca(frame):
    data_x = frame.drop('target', axis=1)
    pca = PCA(n_components=10, copy=True, whiten=True, svd_solver='auto').fit(data_x)
    ratio = pca.explained_variance_ratio_
    print('top 10 components explained %.2f variance' % np.sum(ratio))
    acc = np.array([np.sum(ratio[:i + 1]) * 100 for i in range(len(ratio))])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(range(len(ratio)), ratio)
    ax2 = ax1.twinx()
    ax2.plot(acc, lw=1, c='b')
    ax1.set_title('Principal Component Analysis')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax2.set_ylabel('Accumulated Variance Explained Ratio(%)')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 100)
    plt.show()


def dump(frame):
    dump_path = 'E:/Pycharm/Mywork/aqi/data/4.feature/featured-53'
    frame.to_csv(dump_path+'.csv')
    frame.to_pickle(dump_path+'.pkl')
    print('dumped !')


def get_pearson(frame):
    # calculate pearson coefficients
    corr = frame.corr()
    corr.sort_values(["target"], ascending=False, inplace=True)
    print(corr.target)
    corr.to_csv('E:/Pycharm/Mywork/aqi/log/FE/pearson-featured-53.csv')


if __name__ == "__main__":
    path = 'E:/Pycharm/Mywork/aqi/data/3.offshelf/aqi+54511-4.xlsx'
    data = pd.read_excel(path)
    data = data.astype({'year': 'category', 'month': 'category', 'weekday': 'category', 'class-bj': 'category',
                        'class-lf': 'category', 'class-ts': 'category', 'class-tj': 'category',
                        'class-zjk': 'category', 'class-cd': 'category', 'class-bd': 'category',
                        'PRE-b': 'category', 'class-pm2.5': 'category'})
    data1 = drop(data)
    features = extract(data1)
    get_pearson(features)
    df = transform(features)
    # get_pearson(df)
    print(df.shape)
    dump(df)
    # stability(df)
    pca(df)
