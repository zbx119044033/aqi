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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import xgboost as xgb
import math
from sklearn.metrics import mean_squared_error as mse



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


def loading():
    path = 'E:/Pycharm/Mywork/hujun/samples.xlsx'
    data = pd.read_excel(path)
    data_y = data['target']
    data_x = data.drop('target', axis=1)
    return data_x, data_y


def train_xgb(data_x, data_y):
    # xgboost 不需要归一化
    train_x, temp_x, train_y, temp_y = train_test_split(data_x, data_y, test_size=0.3, shuffle=False, stratify=None)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    # 训练
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)
    dtest = xgb.DMatrix(test_x, label=test_y)
    evallist = [(dval, 'eval'), (dtrain, 'train')]
    param = {'booster': 'gbtree', 'silent': 1, 'nthread': -1,
             'objective': 'reg:linear', 'eval_metric': 'rmse',
             'eta': 0.01, 'gamma': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'lambda': 0.1, 'alpha': 0, 'tree_method': 'auto', 'predictor': 'cpu_predictor',
             }
    bst = xgb.train(param, dtrain, num_boost_round=500, evals=evallist, early_stopping_rounds=10, verbose_eval=20)
    # 反归一化
    train_y_pred = bst.predict(dtrain)
    val_y_pred = bst.predict(dval)
    test_y_pred = bst.predict(dtest)
    '''
    train_y_pred = np.expm1(train_y_pred)
    train_y = np.expm1(train_y)
    val_y_pred = np.expm1(val_y_pred)
    val_y = np.expm1(val_y)
    test_y_pred = np.expm1(test_y_pred)
    test_y = np.expm1(test_y)

    test_y_pred = np.concatenate((val_y_pred, test_y_pred), axis=0)
    test_y = np.concatenate((val_y, test_y), axis=0)
    include = pd.DataFrame(data={'train_y': train_y, 'train_y_pred': train_y_pred})
    include.to_excel('E:/Pycharm/Mywork/aqi/log/result/train.xlsx')
    exclude = pd.DataFrame(data={'test_y': test_y, 'test_y_pred': test_y_pred})
    exclude.to_excel('E:/Pycharm/Mywork/aqi/log/result/test.xlsx')
    '''
    importance(bst, 24)
    xgb.plot_tree(bst, fmap='Gradient boosting tree', num_trees=0, rankdir='UT', ax=None)
    inst = xgb.to_graphviz(bst, fmap='', num_trees=0, rankdir='UT', yes_color='#0000FF', no_color='#FF0000')
    inst.render()
    rmse = math.sqrt(mse(train_y, train_y_pred))
    r2 = r2_score(train_y, train_y_pred)
    print('Train RMSE = %f, R^2 = %f' % (rmse, r2))
    rmse = math.sqrt(mse(test_y, test_y_pred))
    r2 = r2_score(test_y, test_y_pred)
    print('Test RMSE = %f, R^2 = %f' % (rmse, r2))
    drawing(test_y, test_y_pred)
    # bst.save_model('E:/Pycharm/Mywork/aqi/model/xgb-single-53.model')


def drawing(x, y):
    # to draw training curve
    plt.figure()
    plt.scatter(x, y, c='b', marker='.')
    plt.plot([1, 250, 500], [1, 250, 500], 'k--', linewidth=1)
    # plt.text(1, 460, 'RMSE = %.6f\nR^2 = %.6f' % (52.301720, 0.152252), fontsize=10, horizontalalignment='left')
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title("XgBoost prediction via linear regression")
    plt.axis("tight")
    plt.show()


def importance(model, n_features):
    d = model.get_score(importance_type='gain')  # weight, gain, cover
    ss = sorted(d.items(), key=itemgetter(1), reverse=True)
    print(len(ss))
    names = [ss[i][0] for i in range(len(ss))]
    values = [d[name] for name in names]
    frame = pd.DataFrame(data=values, index=names)
    # frame.to_excel('E:/Pycharm/Mywork/aqi/log/FE/importance.xlsx')
    top_names = [ss[i][0] for i in range(n_features)]
    plt.figure(figsize=(11, 9))
    plt.barh(range(n_features), [d[name] for name in top_names], color="b", align="center", height=0.8)
    plt.ylim(-1, n_features)
    plt.yticks(range(n_features), top_names, rotation=0)
    plt.title("Feature importances")
    plt.xlabel('weight')
    plt.ylabel('feature name')
    plt.show()


def count():
    path = 'E:/Pycharm/Mywork/aqi/data/3.offshelf/aqi+54511-4.xlsx'
    data = pd.read_excel(path)
    data_y = data['pm2.5-bj']
    c = len(data_y)
    a = 0  # gb
    b = 0  # who
    for i in data_y:
        if i >= 75:
            a += 1
            b += 1
        elif i >= 25:
            b += 1
        else:
            continue
    print(a, b, c)
    print("over rate for gb = %.2f" % (a/c))
    print("over rate for who = %.2f" % (b/c))


if __name__ == '__main__':
    count()
