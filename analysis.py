# -*- coding:utf-8 -*-

import pickle
import numpy as np
from functional import seq
from matplotlib import pyplot as plt
from operator import itemgetter
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib


def linear_fit():
    with open('E:/Pycharm/Mywork/aqi/data/4.feature/featured-32.pkl', 'rb') as fr:
        data = pickle.load(fr)
    pm = data['pm'].values
    pm1 = data['pm2.5-bj-1'].values
    drop_id = []
    for i in range(len(pm)):
        if abs(pm[i] - pm1[i]) > 250:
            drop_id.append(i)
    data = data.drop(drop_id)

    data_x = pd.concat([data['pm2.5-bj-1'], data['no2-bj-1'], data['pm2.5-lf-1'], data['aqi-bj-1'], data['no2-cd-1'],
                        data['pm10-bj-1'], data['aqi-lf-1'], data['no2-zjk-1'], data['co-bj-1'], data['co-cd-1']], axis=1)
    data_y = data['pm']

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    scaler = StandardScaler()
    train_x.loc[:, :] = scaler.fit_transform(train_x.loc[:, :])
    test_x.loc[:, :] = scaler.transform(test_x.loc[:, :])

    regr = linear_model.LinearRegression()
    regr.fit(train_x , train_y)
    y_pred = regr.predict(test_x)
    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
    print('R^2 score: %.2f' % r2_score(test_y, y_pred))


def pca(frame):
    data_x = frame.drop('pm', axis=1)
    pca = PCA(n_components=10, copy=True, whiten=True, svd_solver='auto').fit(data_x)
    ratio = pca.explained_variance_ratio_
    acc = np.array([np.sum(ratio[:i + 1]) for i in range(len(ratio))])
    plt.figure()
    plt.bar(list(range(len(ratio))), ratio)
    plt.plot(acc, lw=1, c='r')
    plt.title('Principal Component Analysis')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained(%)')
    # plt.ylim(0, 1)
    plt.show()


def split():
    # data set split for stacking
    with open('E:/Pycharm/Mywork/aqi/data/4.feature/featured-53.pkl', 'rb') as fr:
        data = pickle.load(fr)
    train = data.iloc[:1040, :]
    test = data.iloc[1040:, :]
    train1 = train.iloc[:260, :]
    train2 = train.iloc[260:520, :]
    train3 = train.iloc[520:780, :]
    train4 = train.iloc[780:1040, :]
    print(data.shape, train.shape, test.shape, train1.shape, train2.shape, train3.shape, train4.shape)
    dset1 = pd.concat([train2, train3, train4, train1], axis=0, ignore_index=True)
    dset2 = pd.concat([train1, train3, train4, train2], axis=0, ignore_index=True)
    dset3 = pd.concat([train1, train2, train4, train3], axis=0, ignore_index=True)

    dump_path = 'E:/Pycharm/Mywork/aqi/data/4.feature/split/'
    test.to_csv(dump_path + 'test.csv')
    test.to_pickle(dump_path + 'test.pkl')
    dset1.to_csv(dump_path + 'train-1.csv')
    dset1.to_pickle(dump_path + 'train-1.pkl')
    dset2.to_csv(dump_path + 'train-2.csv')
    dset2.to_pickle(dump_path + 'train-2.pkl')
    dset3.to_csv(dump_path + 'train-3.csv')
    dset3.to_pickle(dump_path + 'train-3.pkl')
    train.to_csv(dump_path + 'train-4.csv')
    train.to_pickle(dump_path + 'train-4.pkl')
    print('dumped !')


def mlp_data():
    # data set for GA-MLP training
    features = ['pm', 'day', 'aqi-bj-1', 'pm2.5-bj-1', 'pm10-bj-1', 'no2-bj-1', 'so2-bj-1', 'o3-bj-1', 'aqi-lf-1',
                'no2-lf-1', 'so2-lf-1', 'o3-lf-1', 'aqi-ts-1', 'no2-ts-1', 'so2-ts-1', 'o3-ts-1', 'pm2.5-tj-1',
                'pm10-tj-1', 'co-tj-1', 'no2-tj-1', 'so2-tj-1', 'o3-tj-1', 'aqi-zjk-1', 'pm2.5-zjk-1', 'pm10-zjk-1',
                'co-zjk-1', 'no2-zjk-1', 'so2-zjk-1', 'o3-zjk-1', 'aqi-cd-1', 'pm2.5-cd-1', 'pm10-cd-1', 'co-cd-1',
                'no2-cd-1', 'so2-cd-1', 'o3-cd-1', 'EVP', 'GST-mean', 'GST-max', 'GST-min', 'PRS-mean', 'PRS-max',
                'PRS-min', 'RHU-mean', 'RHU-min', 'SSD', 'TEM-max', 'TEM-min', 'WIN-mean', 'WIN-max', 'WD-max',
                'WIN-ext', 'WD-ext', 'WD-max-sint', 'WIN-max-sin', 'WIN-max-cos', 'WIN-ext-sin', 'WIN-ext-cos',
                'WIN-ext-sint', 'class-pm_1', 'class-pm_2', 'class-pm_3', 'class-pm_4', 'class-pm_5', 'class-pm_6',
                'pollute-lf-1']

    with open('E:/Pycharm/Mywork/aqi/data/4.feature/featured-33.pkl', 'rb') as fr:
        data = pickle.load(fr)
    data_set = data.loc[:, features]
    print(data_set.shape)
    data_set.to_csv('E:/Pycharm/Mywork/aqi/data/4.feature/mlp-data.csv')
    data_set.to_pickle('E:/Pycharm/Mywork/aqi/data/4.feature/mlp-data.pkl')
    print('job done!')


def density(x, y):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    line = np.polyfit(x, y, 1)
    fitting = np.poly1d(line)
    # print(fitting)
    # ss = 'y = 0.89(+0.01)x + 4.23(+1.31)\nR^2 = 0.87\nMPE = 23.96\nRMSE = 23.37'
    sns.set(style='white')
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.plot([-0.5, 450], [fitting(-0.5), fitting(450)], '-b', lw=1)
    plt.plot([-0.5, 450], [-0.5, 450], '--k', lw=1)
    ax.scatter(x, y, c=z, s=30, edgecolor='', cmap='jet')
    # plt.text(10, 300, ss, fontsize=20)
    # plt.xlabel('Observed Values (μg/m3)', fontsize=35, fontproperties='Times New Roman')
    # plt.ylabel('Predicted Values (μg/m3)', fontsize=35, fontproperties='Times New Roman')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.tick_params(direction='out', width=2, length=6)
    # plt.show()
    plt.savefig('E:/Pycharm/Mywork/aqi/log/result/lasso-test-3.png', box_inches='tight', dpi=400)


def evaluate():
    path = 'E:/Pycharm/Mywork/aqi/log/result/result.xlsx'
    data = pd.read_excel(path)

    rmse = math.sqrt(mse(data['train_y'], data['train_y_pred']))
    r2 = r2_score(data['train_y'], data['train_y_pred'])
    mae = mean_absolute_error(data['train_y'], data['train_y_pred'])
    mape = mean_absolute_error(data['train_y'] / data['train_y'], data['train_y_pred'] / data['train_y'])
    ia = 1 - np.sum((data['train_y_pred'] - data['train_y'])**2) / np.sum((np.absolute(data['train_y_pred'] - np.mean(data['train_y'])) + np.absolute(data['train_y'] - np.mean(data['train_y'])))**2)
    bias = np.mean(data['train_y_pred'] - data['train_y'])
    print('Train RMSE = %f, R^2 = %f, MAE = %f, MAPE = %f, IA = %f, Bias = %f' % (rmse, r2, mae, mape, ia, bias))
    density(data['train_y'], data['train_y_pred'])
    '''
    rmse = math.sqrt(mse(data['test_y'], data['test_y_pred']))
    r2 = r2_score(data['test_y'], data['test_y_pred'])
    mae = mean_absolute_error(data['test_y'], data['test_y_pred'])
    mape = mean_absolute_error(data['test_y'] / data['test_y'], data['test_y_pred'] / data['test_y'])
    ia = 1 - np.sum((data['test_y_pred'] - data['test_y'])**2) / np.sum((np.absolute(data['test_y_pred'] - np.mean(data['test_y'])) + np.absolute(data['test_y'] - np.mean(data['test_y'])))**2)
    bias = np.mean(data['test_y_pred'] - data['test_y'])
    print('Test RMSE = %f, R^2 = %f, MAE = %f, MAPE = %f, IA = %f, Bias = %f' % (rmse, r2, mae, mape, ia, bias))
    density(data['test_y'], data['test_y_pred'])
    '''


def simulate():
    # model = xgb.Booster(model_file='E:/Pycharm/Mywork/aqi/model/xgb-54.model')
    with open('E:/Pycharm/Mywork/aqi/model/ensemble-31.pkl', 'rb') as fr:
        model = pickle.load(fr)
    with open('E:/Pycharm/Mywork/aqi/data/4.feature/split/meta-test-31.pkl', 'rb') as fr:
        data_test = pickle.load(fr)
        test_y = data_test['pm']
        test_x = data_test.drop('pm', axis=1)

    with open('E:/Pycharm/Mywork/aqi/data/4.feature/split/meta-train-31.pkl', 'rb') as fr:
        data_train = pickle.load(fr)
        data_y = data_train['pm']
        data_x = data_train.drop('pm', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    scaler = StandardScaler()
    numerical_features = data_x.dtypes[data_x.dtypes != 'uint8'].index
    x_train.loc[:, numerical_features] = scaler.fit_transform(x_train.loc[:, numerical_features])
    test_x.loc[:, numerical_features] = scaler.transform(test_x.loc[:, numerical_features])

    # test_x = xgb.DMatrix(test_x)
    test_y_pred = model.predict(test_x)
    test_y_pred = np.expm1(test_y_pred)
    test_y = np.expm1(test_y)
    include = pd.DataFrame(data={'test_y': test_y, 'test_y_pred': test_y_pred})
    include.to_excel('E:/Pycharm/Mywork/aqi/log/result/train.xlsx')
    print('simulate completed!')


def process():
    path = 'E:/Pycharm/Mywork/aqi/data/4.feature/split/meta-train-32.xlsx'
    data = pd.read_excel(path)
    data = data.astype({'class-pm': 'category'})
    numerical = data.dtypes[data.dtypes != 'category'].index
    print(list(numerical))
    skewness = data[numerical].apply(lambda x: stats.skew(x))
    skewed_features = skewness[abs(skewness) > 0.8].index
    data[skewed_features] = np.log1p(data[skewed_features])
    print("log transformed %d skewed numerical features" % skewness.shape[0])
    print("dummy encoded %d categorical features" % len(data.dtypes[data.dtypes == 'category'].index))
    data = pd.get_dummies(data)
    print(data.dtypes)
    dump_path = 'E:/Pycharm/Mywork/aqi/data/4.feature/split/meta-train-32'
    data.to_csv(dump_path + '.csv')
    data.to_pickle(dump_path + '.pkl')
    print('dumped !')


def importance():
    path = 'E:/Pycharm/Mywork/aqi/log/FE/importance.xlsx'
    data = pd.read_excel(path)
    n_features = 25
    names = data['features'].head(n_features)
    values = data['coefs'].head(n_features)
    colors = data['color'].head(n_features)
    sns.set(style='white')
    plt.figure(figsize=(12, 9))
    plt.barh(range(n_features), values, align="center", height=0.8, color=colors)
    # color=['r', 'b', 'y', 'c', 'g', 'orange', 'brown', 'olive', 'm'])
    plt.ylim(-1, n_features)
    plt.yticks(range(n_features), names, rotation=0)
    plt.title("Feature importance given by averaged scores")
    plt.xlabel('Scores')
    plt.ylabel('Features')
    # plt.show()
    plt.savefig('E:/Pycharm/Mywork/aqi/log/FE/importance-total-3.png', box_inches='tight', dpi=300)


def p_value():
    with open('E:/Pycharm/Mywork/aqi/data/4.feature/featured-53.pkl', 'rb') as fr:
        data = pickle.load(fr)
    data = data.drop(['class-pm2.5_1', 'class-pm2.5_2', 'class-pm2.5_3', 'class-pm2.5_4', 'class-pm2.5_5', 'class-pm2.5_6'], axis=1)
    data_y = data['target']
    data_x = data.drop('target', axis=1)
    selector = SelectKBest(mutual_info_regression, k = 70)  # f_regression, mutual_info_regression
    selector.fit(data_x, data_y)
    scores = selector.scores_
    pvalue = selector.pvalues_
    df = pd.DataFrame({'name': data_x.columns.values, 'socre': scores, 'p-value': pvalue})
    df.to_excel('E:/Pycharm/Mywork/aqi/log/FE/importance.xlsx')
    print('dumped !')


def confusion():
    # to draw normalized confusion matrix
    path = 'E:/Pycharm/Mywork/aqi/log/result/result.xlsx'
    data = pd.read_excel(path)
    test_y = sparse(data['test_y'])
    y_pred = sparse(data['test_y_pred'])
    cm = confusion_matrix(test_y, y_pred)
    print(cm)
    classes = np.array(['I', 'II', 'III', 'IV', 'V/VI'])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Blues
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print(cm)
    fmt = '.2f'
    thresh = cm.max() / 1.3
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.title("Normalized confusion matrix of Ensemble model")
    plt.ylabel('Observed class')
    plt.xlabel('Predicted class')
    plt.show()
    # plt.savefig('E:/Pycharm/Mywork/aqi/log/result/cm-ensemble-1.png', box_inches='tight', dpi=400)


def sparse(serise):
    rt = []
    for i in serise.values:
        if i <= 50:
           rt.append(1)
        elif i < 100:
            rt.append(2)
        elif i < 150:
            rt.append(3)
        elif i < 200:
            rt.append(4)
        # elif i < 301:
        #     rt.append(5)
        else:
            rt.append(5)
    return rt


def lines():
    # to draw normalized confusion matrix
    path = 'E:/Pycharm/Mywork/aqi/data/1.raw/AQI/hour-divided.xlsx'
    data = pd.read_excel(path)
    # pm25 = data['pm2.5']
    # pm10 = data['pm10']
    x = data['hour']
    y = data[['spring', 'summer', 'fall', 'winter']]
    spring = data['spring']
    summer = data['summer']
    autumn = data['fall']
    winter = data['winter']
    # order = data['order']
    mean = [57.83 for i in range(len(spring))]
    base = [25 for i in range(len(spring))]
    sns.set(style='white')
    plt.figure()
    # plt.plot(order, pm25, 'b-', lw = 0.3)
    # plt.scatter(order, pm25, c=data['class'], s=15, cmap='jet')
    # sns.pairplot(data[['pm2.5', 'order', 'class']], size=2.5, kind='scatter', hue='class', plot_kws={'s': 30},
    #              palette=sns.hls_palette(6, l=.3, s=.8),
    #              vars=['pm2.5', 'order'])
    # sns.pointplot(x, y, data=data[['pm2.5', 'order', 'class']], hue='class', scale=0.4, join=True,
    #              palette=sns.hls_palette(6, l=.3, s=.8))
    sns.pointplot(x, spring, scale=0.5, join=True, color='g', markers='o', label='spring')
    sns.pointplot(x, summer, scale=0.5, join=True, color='b', markers='*', label='summer')
    sns.pointplot(x, autumn, scale=0.5, join=True, color='y', markers='^', label='autumn')
    sns.pointplot(x, winter, scale=0.5, join=True, color='purple', markers='+', label='winter')
    plt.plot(x, mean, 'r--', lw = 2, label='mean')
    plt.plot(x, base, 'k--', lw = 2, label='WHO Guideline')
    # plt.legend()
    # plt.show()
    plt.savefig('E:/Pycharm/Mywork/aqi/log/EDA/PM25/hour.png', box_inches='tight', dpi=300)


def curve():
    path = 'E:/PyCharm/mywork/aqi/log/result/drawing.xlsx'
    data = pd.read_excel(path)
    sns.set(style='white')
    plt.figure()
    lw = 1
    plt.plot(data['time'], data['observation'], 'b-', lw=lw, label='observation')
    plt.plot(data['time'], data['lasso'], 'g--', lw=lw, label='Lasso')
    plt.plot(data['time'], data['svr'], 'y--', lw=lw, label='SVR')
    plt.plot(data['time'], data['xgboost'], 'c--', lw=lw, label='XGBoost')
    plt.plot(data['time'], data['adaboost'], 'm--', lw=lw, label='Adaboost')
    plt.plot(data['time'], data['mlp'], 'r--', lw=lw, label='GA-MLP')
    plt.plot(data['time'], data['ensemble'], 'k--', lw=lw, label='Ensemble')
    plt.legend()
    plt.show()
    # plt.savefig('E:/PyCharm/mywork/aqi/log/result/comparison-1.png', box_inches='tight', dpi=300)


if __name__ == '__main__':
    # linear_fit()
    # pca()
    # split()
    # mlp_data()
    # density()
    # simulate()
    # evaluate()
    # process()
    # importance()
    # p_value()
    # confusion()
    # lines()
    curve()
