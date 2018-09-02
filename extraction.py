# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import boxcox1p
from sklearn.linear_model import RandomizedLasso
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def extract(frame):
    # 采用origin将所有变量拟合之后以拟合公式为新特征的提取方式，存在信息泄露风险
    # featured-4x
    # 1* Simplifications of existing features
    frame["SimpClass-cd-1"] = frame['class-cd-1'].replace({1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}).astype('category')
    frame["SimpClass-lf-1"] = frame['class-lf-1'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}).astype('category')
    frame["SimpClass-tj-1"] = frame['class-tj-1'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}).astype('category')
    frame["SimpClass-ts-1"] = frame['class-ts-1'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}).astype('category')
    frame["SimpClass-zjk-1"] = frame['class-zjk-1'].replace({1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}).astype('category')

    # 2* Polynomials on the top 10 existing features
    # pm10-bj-1
    frame["pm10-bj-1-poly"] = 3.2 * frame["pm10-bj-1"] ** 0.69258
    # co-bj-1
    frame["co-bj-1-poly"] = 72.44 * frame["co-bj-1"] ** 0.62788
    # so2-bj-1
    frame["so2-bj-1-poly"] = 34.11 * frame["so2-bj-1"] ** 0.35093
    # o3-bj-1
    frame["o3-bj-1-poly"] = 174.87 - 2.966 * frame["o3-bj-1"] + 0.02 * frame["o3-bj-1"] ** 2 - 4e-5 * frame["o3-bj-1"] ** 3
    # so2-lf-1
    frame["so2-lf-1-poly"] = 23.87 * frame["so2-lf-1"] ** 0.39
    # so2-tj-1
    frame["pm10-tj-1-poly"] = 16.26 + 0.58 * frame["pm10-tj-1"] - 5.22e-4 * frame["pm10-tj-1"] ** 2
    # no2-cd-1
    frame["no2-cd-1-poly"] = 62.49 - 3.72 * frame["no2-cd-1"] + 0.15 * frame["no2-cd-1"] ** 2 - 0.0011 * frame["no2-cd-1"] ** 3
    # so2-cd-1
    frame["so2-cd-1-poly"] = 13.34 + 3.44 * frame["so2-cd-1"] - 0.02214 * frame["so2-cd-1"] ** 2
    # no2-bd-1
    frame["no2-bd-1-poly"] = 71.5 - 1.61 * frame["no2-bd-1"] + 0.036 * frame["no2-bd-1"] ** 2 - 1.38678e-4 * frame["no2-bd-1"] ** 3
    # so2-ts-1
    frame['so2-ts-1-poly'] = 33.83 + 0.94 * frame['so2-ts-1'] - 0.00194 * frame['so2-ts-1'] ** 2
    '''
    # pollute-bj-1
    frame["pollute-bj-1-s2"] = frame["pollute-bj-1"] ** 2
    frame["pollute-bj-1-s3"] = frame["pollute-bj-1"] ** 3
    frame["pollute-bj-1-sq"] = np.sqrt(frame["pollute-bj-1"])
    # pollute-lf-1
    frame["pollute-lf-1-s2"] = frame["pollute-lf-1"] ** 2
    frame["pollute-lf-1-s3"] = frame["pollute-lf-1"] ** 3
    frame["pollute-lf-1-sq"] = np.sqrt(frame["pollute-lf-1"])
    # no2Grd-bj-1
    frame["no2Grd-bj-1-s2"] = frame["no2Grd-bj-1"] ** 2
    frame["no2Grd-bj-1-s3"] = frame["no2Grd-bj-1"] ** 3
    frame["no2Grd-bj-1-sq"] = np.sqrt(frame["no2Grd-bj-1"])
    # no2Scr-bj-1
    frame["no2Scr-bj-1-s2"] = frame["no2Scr-bj-1"] ** 2
    frame["no2Scr-bj-1-s3"] = frame["no2Scr-bj-1"] ** 3
    frame["no2Scr-bj-1-sq"] = np.sqrt(frame["no2Scr-bj-1"])
    # OvallGrd-bj-1
    frame["OvallGrd-bj-1-s2"] = frame["OvallGrd-bj-1"] ** 2
    frame["OvallGrd-bj-1-s3"] = frame["OvallGrd-bj-1"] ** 3
    frame["OvallGrd-bj-1-sq"] = np.sqrt(frame["OvallGrd-bj-1"])
    # pm25Grd-bj-1
    frame["pm25Grd-bj-1-s2"] = frame["pm25Grd-bj-1"] ** 2
    frame["pm25Grd-bj-1-s3"] = frame["pm25Grd-bj-1"] ** 3
    frame["pm25Grd-bj-1-sq"] = np.sqrt(frame["pm25Grd-bj-1"])
    # pm10Grd-bj-1
    frame["pm10Grd-bj-1-s2"] = frame["pm10Grd-bj-1"] ** 2
    frame["pm10Grd-bj-1-s3"] = frame["pm10Grd-bj-1"] ** 3
    frame["pm10Grd-bj-1-sq"] = np.sqrt(frame["pm10Grd-bj-1"])
    # pm2.5-lf-1
    frame["pm2.5-lf-1-s2"] = frame["pm2.5-lf-1"] ** 2
    frame["pm2.5-lf-1-s3"] = frame["pm2.5-lf-1"] ** 3
    frame["pm2.5-lf-1-sq"] = np.sqrt(frame["pm2.5-lf-1"])
    # no2-cd-1
    frame["no2-cd-1-s2"] = frame["no2-cd-1"] ** 2
    frame["no2-cd-1-s3"] = frame["no2-cd-1"] ** 3
    frame["no2-cd-1-sq"] = np.sqrt(frame["no2-cd-1"])
    # aqi-bj-1
    frame["aqi-bj-1-s2"] = frame["aqi-bj-1"] ** 2
    frame["aqi-bj-1-s3"] = frame["aqi-bj-1"] ** 3
    frame["aqi-bj-1-sq"] = np.sqrt(frame["aqi-bj-1"])
    # pm2.5-bj-1
    frame["pm2.5-bj-1-s2"] = frame["pm2.5-bj-1"] ** 2
    frame["pm2.5-bj-1-s3"] = frame["pm2.5-bj-1"] ** 3
    frame["pm2.5-bj-1-sq"] = np.sqrt(frame["pm2.5-bj-1"])
    # no2-bj-1
    frame["no2-bj-1-s2"] = frame["no2-bj-1"] ** 2
    frame["no2-bj-1-s3"] = frame["no2-bj-1"] ** 3
    frame["no2-bj-1-sq"] = np.sqrt(frame["no2-bj-1"])
    # aqi-lf-1
    frame["aqi-lf-1-s2"] = frame["aqi-lf-1"] ** 2
    frame["aqi-lf-1-s3"] = frame["aqi-lf-1"] ** 3
    frame["aqi-lf-1-sq"] = np.sqrt(frame["aqi-lf-1"])
    # co-lf-1
    frame["co-lf-1-poly"] = 64.48 * frame["co-lf-1"] ** 0.67
    # pm10-lf-1
    frame["pm10-lf-1-s2"] = frame["pm10-lf-1"] ** 2
    frame["pm10-lf-1-s3"] = frame["pm10-lf-1"] ** 3
    frame["pm10-lf-1-poly"] = 2.28 * frame["pm10-lf-1"] ** 0.725
    # no2-zjk-1
    frame["no2-zjk-1-s2"] = frame["no2-zjk-1"] ** 2
    frame["no2-zjk-1-s3"] = frame["no2-zjk-1"] ** 3
    frame["no2-zjk-1-sq"] = np.sqrt(frame["no2-zjk-1"])
    '''

    # 3* nonlinear extraction on meteorological features
    mon = frame.loc[:, 'month'].astype('int32')
    frame['month-sin'] = 80.88 + 25.11 * np.sin((mon + 4.15) / 6.92 * np.pi)
    frame["WD-max-sin"] = abs(np.sin((frame["WD-max"] - 1) * 1 / 8 * np.pi))
    frame["WD-max-cos"] = abs(np.cos((frame["WD-max"] - 1) * 1 / 8 * np.pi))
    frame["WD-max-sint"] = abs(np.sin((frame["WD-max"] - 1) * 1 / 8 * np.pi + 4 / 3 * np.pi))
    frame["WD-max-sine"] = 54.15 + 51.454 * np.sin((frame["WD-max"] - 26.62) / 12.64 * np.pi)
    frame["WD-ext-sin"] = abs(np.sin((frame["WD-ext"] - 1) * 1 / 8 * np.pi))
    frame["WD-ext-cos"] = abs(np.cos((frame["WD-ext"] - 1) * 1 / 8 * np.pi))
    frame["WD-ext-sint"] = abs(np.sin((frame["WD-ext"] - 1) * 1 / 8 * np.pi + 4 / 3 * np.pi))
    frame["WD-ext-sine"] = 30.63 - 72.41 * np.sin((frame["WD-ext"] + 16.88) / 16.556 * np.pi)

    frame["o3-bj-1-exp"] = 63.55 + 127.33 * np.exp((2.729 - frame["o3-bj-1"]) / 19.77)
    frame["o3-lf-1-exp"] = 66.18 + 140.09 * np.exp(- frame["o3-lf-1"] / 15.48)
    frame["so2-tj-1-exp"] = 122.15 - 98.09 * np.exp(- frame["so2-tj-1"] / 29.42)
    frame["o3-tj-1-exp"] = 66.71 + 162.94 * np.exp(- frame["o3-tj-1"] / 15.71)
    frame["o3-zjk-1-exp"] = 64.73 + 266.22 * np.exp(- frame["o3-zjk-1"] / 23.4)
    frame["o3-cd-1-exp"] = 64.7 + 158.85 * np.exp(- frame["o3-cd-1"] / 24.11)
    frame["o3-bd-1-exp"] = 66.32 + 223.74 * np.exp(- frame["o3-bd-1"] / 13.83)
    frame["o3-ts-1-exp"] = 67.23 + 122.52 * np.exp(- frame["o3-ts-1"] / 19.66)
    frame['EVP-exp'] = 50.8 + 129.24 * np.exp(-frame['EVP'] / 16.98)
    frame['WIN-mean-exp'] = 34.42 + 293.49 * np.exp(-frame['WIN-mean'] / 9.6)
    frame['WIN-max-exp'] = 38.02 + 467.11 * np.exp(-frame['WIN-max'] / 16.64)
    frame['WIN-ext-exp'] = 34.99 + 373.63 * np.exp(-frame['WIN-ext'] / 32.035)

    frame['pm2.5-zjk-1-log'] = -90.27 + 50.6 * np.log(frame['pm2.5-zjk-1'] + 0.35)
    frame['pm10-zjk-1-log'] = -105.64 + 43.26 * np.log(frame['pm10-zjk-1'] + 2.6)
    frame['so2-zjk-1-log'] = -66.73 + 41.88 * np.log(frame['so2-zjk-1'] + 3.92)
    frame['pm2.5-cd-1-log'] = -249.51 + 80.4 * np.log(frame['pm2.5-cd-1'] + 20.7)
    frame['pm10-cd-1-log'] = -251.38 + 70.38 * np.log(frame['pm10-cd-1'] + 25.575)
    frame['so2-bd-1-log'] = -46.88 + 33.66 * np.log(frame['so2-bd-1'] + 1.49566)
    frame['co-bd-1-hill'] = 258.52 * frame['co-bd-1'] ** 0.77611 / (frame['co-bd-1'] ** 0.77611 + 4.58527 ** 0.77611)

    # 4* Combinations of existing features
    frame['WIN-max-project1'] = frame['WIN-max-exp'] * frame["WD-max-sint"]
    frame['WIN-max-project2'] = frame['WIN-max-exp'] * frame["WD-max-sine"]
    frame['WIN-ext-project1'] = frame['WIN-ext-exp'] * frame["WD-ext-sint"]
    frame['WIN-ext-project2'] = frame['WIN-ext-exp'] * frame["WD-ext-sine"]
    frame["WIN-max-sin"] = frame["WIN-max"] * frame["WD-max-sin"]
    frame["WIN-max-cos"] = frame["WIN-max"] * frame["WD-max-cos"]
    frame["WIN-max-sint"] = frame["WIN-max"] * frame["WD-max-sint"]
    frame["WIN-max-sine"] = frame["WIN-max"] * frame["WD-max-sine"]
    frame["WIN-ext-sin"] = frame["WIN-ext"] * frame["WD-ext-sin"]
    frame["WIN-ext-cos"] = frame["WIN-ext"] * frame["WD-ext-cos"]
    frame["WIN-ext-sint"] = frame["WIN-ext"] * frame["WD-ext-sint"]
    frame["WIN-ext-sine"] = frame["WIN-ext"] * frame["WD-ext-sine"]
    '''
    frame["OvallGrd-bj-1"] = frame["class-bj-1"] * frame["aqi-bj-1"]
    frame["pm25Scr-bj-1"] = frame["aqi-bj-1"] * frame["pm2.5-bj-1"]
    frame["pm25Grd-bj-1"] = frame["class-bj-1"] * frame["pm2.5-bj-1"]
    frame["pm10Scr-bj-1"] = frame["aqi-bj-1"] * frame["pm10-bj-1"]
    frame["pm10Grd-bj-1"] = frame["class-bj-1"] * frame["pm10-bj-1"]
    frame["no2Scr-bj-1"] = frame["aqi-bj-1"] * frame["no2-bj-1"]
    frame["no2Grd-bj-1"] = frame["class-bj-1"] * frame["no2-bj-1"]
    # frame["pollute-bj-1"] = 0.28 * frame["pm2.5-bj-1"] + 0.24 * frame["no2-bj-1"] + 0.04 * frame["pm10-bj-1"] - 0.08 * frame['co-bj-1'] + 0.09 * frame['so2-bj-1'] + 0.09 * frame['o3-bj-1']
    # 廊坊
    frame["OvallGrd-lf-1"] = frame["class-lf-1"] * frame["aqi-lf-1"]
    frame["pm25Scr-lf-1"] = frame["aqi-lf-1"] * frame["pm2.5-lf-1"]
    frame["pm25Grd-lf-1"] = frame["class-lf-1"] * frame["pm2.5-lf-1"]
    frame["pm10Scr-lf-1"] = frame["aqi-lf-1"] * frame["pm10-lf-1"]
    frame["pm10Grd-lf-1"] = frame["class-lf-1"] * frame["pm10-lf-1"]
    # frame["pollute-lf-1"] = 0.22 * frame['pm2.5-bj-1'] + 0.11 * frame['no2-bj-1'] - 0.02 * frame['pm2.5-lf-1'] + 0.14 * frame['aqi-bj-1'] - 0.05 * frame['no2-cd-1'] + 0.09 * frame['pm10-bj-1'] - 0.06 * frame['aqi-lf-1'] + 0.31 * frame['no2-zjk-1'] - 0.19 * frame['co-bj-1'] + 0.1 * frame['co-cd-1']
    '''
    return frame

