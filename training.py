# -*- coding:utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVR, LinearSVR
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from operator import itemgetter
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RandomizedLasso


def loading():
    with open('E:/Pycharm/Mywork/aqi/data/4.feature/featured-53.pkl', 'rb') as fr:
        data = pickle.load(fr)
    data = data.drop(['class-pm2.5_1', 'class-pm2.5_2', 'class-pm2.5_3', 'class-pm2.5_4', 'class-pm2.5_5', 'class-pm2.5_6'], axis=1)
    data_y = data['target']
    data_x = data.drop('target', axis=1)
    return data_x, data_y


def train_linear(data_x, data_y):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    scaler = StandardScaler()
    numerical_features = data_x.dtypes[data_x.dtypes != 'uint8'].index
    train_x.loc[:, numerical_features] = scaler.fit_transform(train_x.loc[:, numerical_features])
    test_x.loc[:, numerical_features] = scaler.transform(test_x.loc[:, numerical_features])

    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.004, 0.006, 0.01, 0.02, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 10],
                    max_iter=1000000, cv=4)
    lasso.fit(train_x, train_y)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = LassoCV(alphas=[alpha * .5, alpha * .55, alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                            alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                            alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4, alpha * 1.45, alpha * 1.5],
                    max_iter=1000000, cv=4)
    lasso.fit(train_x, train_y)
    alpha = lasso.alpha_
    # 反归一化
    train_y_pred = lasso.predict(train_x)
    test_y_pred = lasso.predict(test_x)
    train_y_pred = np.expm1(train_y_pred)
    test_y_pred = np.expm1(test_y_pred)
    train_y = np.expm1(train_y)
    test_y = np.expm1(test_y)
    include = pd.DataFrame(data={'train_y': train_y, 'train_y_pred': train_y_pred})
    include.to_excel('E:/Pycharm/Mywork/aqi/log/result/train.xlsx')
    exclude = pd.DataFrame(data={'test_y': test_y, 'test_y_pred': test_y_pred})
    exclude.to_excel('E:/Pycharm/Mywork/aqi/log/result/test.xlsx')
    # 评估
    train_rmse = math.sqrt(mse(train_y, train_y_pred))
    train_r2 = r2_score(train_y, train_y_pred)
    test_rmse = math.sqrt(mse(test_y, test_y_pred))
    test_r2 = r2_score(test_y, test_y_pred)
    # drawing(test_y, test_y_pred)
    print("Best alpha :", alpha)
    print("Lasso Training set :RMSE = %.2f , R^2 = %.2f" % (train_rmse, train_r2))
    print("Lasso Test set: RMSE = %.2f , R^2 = %.2f" % (test_rmse, test_r2))
    '''
    # Plot residuals
    plt.scatter(train_y_pred, train_y_pred - train_y, c="blue", marker="o", label="Training data")
    plt.scatter(test_y_pred, test_y_pred - test_y, c="lightgreen", marker="o", label="Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=50, xmax=400, color="red")
    plt.show()
    # Plot predictions
    plt.scatter(train_y_pred, train_y, c="blue", marker="o", label="Training data")
    plt.scatter(test_y_pred, test_y, c="lightgreen", marker="o", label="Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([50, 450], [50, 450], c="red")
    plt.show()
    '''
    # Plot important coefficients
    coefs = pd.Series(lasso.coef_, index=train_x.columns)
    frame = pd.DataFrame(data=coefs, index=train_x.columns)
    frame.to_excel('E:/Pycharm/Mywork/aqi/log/FE/importance.xlsx')
    print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " + str(sum(coefs == 0)))
    imp_coefs = pd.concat([coefs.sort_values().head(15), coefs.sort_values().tail(15)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()
    with open('E:/Pycharm/Mywork/aqi/model/lasso-single.pkl', 'wb') as handle:
        pickle.dump(lasso, handle)


def train_svc(data_x, data_y):
    # 归一化
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.25, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    scaler = StandardScaler()
    numerical_features = data_x.dtypes[data_x.dtypes != 'uint8'].index
    train_x.loc[:, numerical_features] = scaler.fit_transform(train_x.loc[:, numerical_features])
    test_x.loc[:, numerical_features] = scaler.transform(test_x.loc[:, numerical_features])
    # 训练
    n = 50
    c = np.linspace(0.5, 20, n)
    scores = []
    scorer = make_scorer(mse, greater_is_better=False)
    for i in range(n):
        reg = LinearSVR(C=c[i], epsilon=.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=False, dual=True,
                        max_iter=50000)
        score = np.sqrt(-cross_val_score(reg, train_x, train_y, scoring=scorer, cv=4)).mean()  # RMSE
        scores.append(score)
        print('item %d completed' % i)
    print(scores)
    plt.figure()
    plt.plot(c, scores, 'b')
    plt.xlabel('c')
    plt.ylabel('RMSE')
    plt.show()


def train_svr(data_x, data_y):
    # 归一化
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    scaler = StandardScaler()
    numerical_features = data_x.dtypes[data_x.dtypes != 'uint8'].index
    train_x.loc[:, numerical_features] = scaler.fit_transform(train_x.loc[:, numerical_features])
    test_x.loc[:, numerical_features] = scaler.transform(test_x.loc[:, numerical_features])
    # 训练
    # clf = SVR(kernel='rbf', C=5.0, gamma=0.0003689655172413793, tol=0.001, epsilon=0.1, shrinking=True)
    reg = LinearSVR(epsilon=.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=False, dual=True, max_iter=50000)
    clf = GridSearchCV(reg, {'C': np.linspace(0.01, 5, 50)}, refit=True, cv=4)
    # clf = LinearSVR(C=0.10110909090909091, epsilon=.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=False, dual=True, max_iter=50000)
    clf.fit(train_x, train_y)
    print('best params: ', clf.best_params_)
    print('best score: ', clf.best_score_)
    # print('mean test score: ', clf.cv_results_['mean_test_score'])
    # 反归一化
    train_y_pred = clf.predict(train_x)
    test_y_pred = clf.predict(test_x)
    train_y = np.expm1(train_y)
    train_y_pred = np.expm1(train_y_pred)
    test_y = np.expm1(test_y)
    test_y_pred = np.expm1(test_y_pred)
    include = pd.DataFrame(data={'train_y': train_y, 'train_y_pred': train_y_pred})
    include.to_excel('E:/Pycharm/Mywork/aqi/log/result/train.xlsx')
    exclude = pd.DataFrame(data={'test_y': test_y, 'test_y_pred': test_y_pred})
    exclude.to_excel('E:/Pycharm/Mywork/aqi/log/result/test.xlsx')
    # 评估
    rmse = math.sqrt(mse(train_y, train_y_pred))
    r2 = r2_score(train_y, train_y_pred)
    print('Train RMSE = %f, R^2 = %f' % (rmse, r2))
    rmse = math.sqrt(mse(test_y, test_y_pred))
    r2 = r2_score(test_y, test_y_pred)
    print('Test RMSE = %f, R^2 = %f' % (rmse, r2))
    # drawing(test_y, test_y_pred)
    with open('E:/Pycharm/Mywork/aqi/model/svc-single.pkl', 'wb') as handle:
        pickle.dump(clf, handle)


def train_rbf(data_x, data_y):
    # 归一化
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    scaler = StandardScaler()
    numerical_features = data_x.dtypes[data_x.dtypes != 'uint8'].index
    train_x.loc[:, numerical_features] = scaler.fit_transform(train_x.loc[:, numerical_features])
    test_x.loc[:, numerical_features] = scaler.transform(test_x.loc[:, numerical_features])
    # Grid search cross validation
    n = 40
    c_range = np.linspace(0.01, 3, n)
    gamma_range = np.linspace(1e-8, 0.001, n)
    scorer = make_scorer(mse, greater_is_better=False)
    reg = SVR(kernel='rbf', tol=0.001, epsilon=0.1, shrinking=True)
    clf = GridSearchCV(reg, {'C': c_range, 'gamma': gamma_range}, scoring=scorer, refit=True, cv=4)
    clf.fit(train_x, train_y)
    print('best params: ', clf.best_params_)
    print('best score: ', clf.best_score_)
    # print('mean test score: ', clf.cv_results_['mean_test_score'])
    # 反归一化
    train_y_pred = clf.predict(train_x)
    test_y_pred = clf.predict(test_x)
    train_y = np.expm1(train_y)
    train_y_pred = np.expm1(train_y_pred)
    test_y = np.expm1(test_y)
    test_y_pred = np.expm1(test_y_pred)
    include = pd.DataFrame(data={'train_y': train_y, 'train_y_pred': train_y_pred})
    include.to_excel('E:/Pycharm/Mywork/aqi/log/result/train.xlsx')
    exclude = pd.DataFrame(data={'test_y': test_y, 'test_y_pred': test_y_pred})
    exclude.to_excel('E:/Pycharm/Mywork/aqi/log/result/test.xlsx')
    # 评估
    rmse = math.sqrt(mse(train_y, train_y_pred))
    r2 = r2_score(train_y, train_y_pred)
    print('Train RMSE = %f, R^2 = %f' % (rmse, r2))
    rmse = math.sqrt(mse(test_y, test_y_pred))
    r2 = r2_score(test_y, test_y_pred)
    print('Test RMSE = %f, R^2 = %f' % (rmse, r2))
    # drawing(test_y, test_y_pred)
    with open('E:/Pycharm/Mywork/aqi/model/ensemble-31.pkl', 'wb') as handle:
        pickle.dump(clf, handle)


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
             'eta': 0.005, 'gamma': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'lambda': 0.1, 'alpha': 0, 'tree_method': 'auto', 'predictor': 'cpu_predictor',
             }
    bst = xgb.train(param, dtrain, num_boost_round=1100, evals=evallist, early_stopping_rounds=10, verbose_eval=20)
    # 反归一化
    train_y_pred = bst.predict(dtrain)
    val_y_pred = bst.predict(dval)
    test_y_pred = bst.predict(dtest)
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

    importance(bst, 25)
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
    bst.save_model('E:/Pycharm/Mywork/aqi/model/xgb-single-53.model')


def train_rf(data_x, data_y):
    # random forest 不需要归一化
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    # 训练
    forest = ExtraTreesRegressor(criterion='mse', max_depth=11, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 bootstrap=True, oob_score=False, n_jobs=4)
    clf = GridSearchCV(forest, {'n_estimators': np.arange(50, 810, 50)}, refit=True, cv=4)
    clf.fit(train_x, train_y)
    # 评估
    train_y_pred = clf.predict(train_x)
    test_y_pred = clf.predict(test_x)
    train_y_pred = np.expm1(train_y_pred)
    train_y = np.expm1(train_y)
    test_y_pred = np.expm1(test_y_pred)
    test_y = np.expm1(test_y)

    include = pd.DataFrame(data={'train_y': train_y, 'train_y_pred': train_y_pred})
    include.to_excel('E:/Pycharm/Mywork/aqi/log/result/train.xlsx')
    exclude = pd.DataFrame(data={'test_y': test_y, 'test_y_pred': test_y_pred})
    exclude.to_excel('E:/Pycharm/Mywork/aqi/log/result/test.xlsx')

    print("best parameters:", clf.best_params_)
    rmse = math.sqrt(mse(train_y, train_y_pred))
    r2 = r2_score(train_y, train_y_pred)
    print('RMSE = %f, R^2 = %f' % (rmse, r2))
    rmse = math.sqrt(mse(test_y, test_y_pred))
    r2 = r2_score(test_y, test_y_pred)
    print('RMSE = %f, R^2 = %f' % (rmse, r2))
    drawing(test_y, test_y_pred)


def train_adaboost(data_x, data_y):
    # random forest 不需要归一化
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    # 训练
    forest = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, max_features='sqrt'), loss='exponential')
    clf = GridSearchCV(forest, {'n_estimators': [100, 200, 230, 260, 290, 320, 350, 380, 410, 440, 470, 500],
                                'learning_rate': [.6, .62, .64, .66, .68, .7, .72, .74, .76, .78, .8, .82, .84, .86, .88, .9]}, refit=True, cv=4)
    clf.fit(train_x, train_y)
    # 评估
    train_y_pred = clf.predict(train_x)
    test_y_pred = clf.predict(test_x)
    train_y_pred = np.expm1(train_y_pred)
    train_y = np.expm1(train_y)
    test_y_pred = np.expm1(test_y_pred)
    test_y = np.expm1(test_y)

    include = pd.DataFrame(data={'train_y': train_y, 'train_y_pred': train_y_pred})
    include.to_excel('E:/Pycharm/Mywork/aqi/log/result/train.xlsx')
    exclude = pd.DataFrame(data={'test_y': test_y, 'test_y_pred': test_y_pred})
    exclude.to_excel('E:/Pycharm/Mywork/aqi/log/result/test.xlsx')

    print("best parameters:", clf.best_params_)
    rmse = math.sqrt(mse(train_y, train_y_pred))
    r2 = r2_score(train_y, train_y_pred)
    print('Train RMSE = %f, R^2 = %f' % (rmse, r2))
    rmse = math.sqrt(mse(test_y, test_y_pred))
    r2 = r2_score(test_y, test_y_pred)
    print('Test RMSE = %f, R^2 = %f' % (rmse, r2))
    # drawing(test_y, test_y_pred)
    # Plot important coefficients
    estimator = clf.best_estimator_
    coefs = pd.Series(estimator.feature_importances_, index=train_x.columns)
    frame = pd.DataFrame(data=estimator.feature_importances_, index=train_x.columns)
    frame.to_excel('E:/Pycharm/Mywork/aqi/log/FE/importance.xlsx')
    imp_coefs = coefs.sort_values().head(30)
    imp_coefs.plot(kind="barh")
    plt.title("Feature importance given by Adaboost Regressor")
    plt.show()
    with open('E:/Pycharm/Mywork/aqi/model/ada-single.pkl', 'wb') as handle:
        pickle.dump(clf, handle)


def grid_search(data_x, data_y):
    # 归一化
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, stratify=None)
    print("X_train : " + str(train_x.shape) + "  X_test : " + str(test_x.shape))
    print("y_train : " + str(train_y.shape) + "  y_test : " + str(test_y.shape))
    scaler = StandardScaler()
    numerical_features = data_x.dtypes[data_x.dtypes != 'uint8'].index
    train_x.loc[:, numerical_features] = scaler.fit_transform(train_x.loc[:, numerical_features])
    test_x.loc[:, numerical_features] = scaler.transform(test_x.loc[:, numerical_features])
    # 网格搜索
    n = 30
    c = np.linspace(0.01, 3, n)
    gamma = np.linspace(1e-8, 0.001, n)
    # n_estimators = np.linspace(50, 850, n)
    # learning_rate = np.linspace(0.01, 1.1, n)
    scores = np.zeros((n, n))
    scorer = make_scorer(mse, greater_is_better=False)
    for i in range(n):
        for j in range(n):
            clf = SVR(C=c[i], kernel='rbf', gamma=gamma[j], tol=0.001, epsilon=0.1, shrinking=True)
            # forest = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7, max_features='sqrt'), loss='exponential',
            #                            n_estimators=int(n_estimators[i]), learning_rate=learning_rate[j])
            score = np.sqrt(-cross_val_score(clf, train_x, train_y, scoring=scorer, cv=4)).mean()  # RMSE
            scores[j, i] = score
            print('item %d completed' % ((i+1)*(j+1)))
    contourline(c, gamma, scores)
    # contourline(n_estimators, learning_rate, scores)


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


def contourline(c, g, y):
    # to draw contour line and y.shape = (len(c), len(g)), 3d->2d
    plt.figure()
    handel = plt.contour(c, g, y)
    plt.clabel(handel, inline=1, fontsize=10)
    plt.title('Grid search for best c, gamma with CV=4, balanced')
    plt.xlabel('parameter C')
    plt.ylabel('parameter gamma')
    plt.show()


def importance(model, n_features):
    d = model.get_score(importance_type='gain')  # weight, gain, cover
    ss = sorted(d.items(), key=itemgetter(1), reverse=True)
    print(len(ss))
    names = [ss[i][0] for i in range(len(ss))]
    values = [d[name] for name in names]
    frame = pd.DataFrame(data=values, index=names)
    frame.to_excel('E:/Pycharm/Mywork/aqi/log/FE/importance.xlsx')
    top_names = [ss[i][0] for i in range(n_features)]
    plt.figure(figsize=(11, 9))
    plt.barh(range(n_features), [d[name] for name in top_names], color="b", align="center", height=0.8)
    plt.ylim(-1, n_features)
    plt.yticks(range(n_features), top_names, rotation=0)
    plt.title("Feature importances")
    plt.xlabel('weight')
    plt.ylabel('feature name')
    plt.show()


def stability(data_x, data_y):
    scaler = StandardScaler()
    numerical_features = data_x.dtypes[data_x.dtypes != 'uint8'].index
    data_x.loc[:, numerical_features] = scaler.fit_transform(data_x.loc[:, numerical_features])
    selection = RandomizedLasso(alpha='bic', scaling=0.8, sample_fraction=0.8, max_iter=100000)
    selection.fit_transform(data_x, data_y)
    coefs = selection.scores_
    frame = pd.DataFrame(data=coefs, index=data_x.columns)
    frame.to_excel('E:/Pycharm/Mywork/aqi/log/FE/importance.xlsx')
    print("Stability picked %d features and eliminated the other %d of all %d" % (sum(coefs != 0), sum(coefs == 0), frame.shape[0]))


if __name__ == '__main__':
    rt_x, rt_y = loading()
    # train_linear(rt_x, rt_y)
    # train_svc(rt_x, rt_y)
    # train_svr(rt_x, rt_y)
    train_xgb(rt_x, rt_y)
    # train_rf(rt_x, rt_y)
    # train_adaboost(rt_x, rt_y)
    # grid_search(rt_x, rt_y)
    # train_rbf(rt_x, rt_y)
    # stability(rt_x, rt_y)
