# -*- coding:utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


def distribution(series, name):  # 绘制分布直方图
    # print(series.describe())  # descriptive statistics summary
    s1 = "max:%.2f" % series.max()
    s2 = "min:%.2f" % series.min()
    s3 = "25%%:%.2f" % series.quantile(0.25)
    s4 = "50%%:%.2f" % series.quantile(0.5)
    s5 = "75%%:%.2f" % series.quantile(0.75)
    s6 = "avg:%.2f" % series.mean()
    s7 = "std:%.2f" % series.std()
    s8 = "skewness: %.2f" % series.skew()  # 偏度
    s9 = "kurtosis: %.2f" % series.kurt()  # 峰度
    des = s1+'\n'+s2+'\n'+s3+'\n'+s4+'\n'+s5+'\n'+s6+'\n'+s7+'\n'+s8+'\n'+s9
    plt.figure()
    sns.set_style('white')
    sns.distplot(series, hist=True, kde=True, fit=norm, rug=True, rug_kws={"color": "g"},
                 kde_kws={"label": "kernel density estimate"}, fit_kws={"label": "estimated normal PDF"},
                 hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1, "color": "b"})
    # sns.plt.text(series.max()*0.8, 0.005, des)
    plt.legend(loc='upper right')
    sns.plt.title('%s Distribution' % name, fontsize=13)
    sns.plt.xlabel(name, fontsize=13)
    sns.plt.ylabel('Frequency', fontsize=13)
    plt.savefig('%s-distplot.png' % name, bbox_inches='tight', dpi=400)
    # sns.plt.show()
    # plt.close()
    plt.figure()
    stats.probplot(series, plot=plt)
    plt.savefig('%s-probability-plot.png' % name, bbox_inches='tight', dpi=400)
    # plt.show()
    # plt.close()


def distribution_log(series, name):  # 绘制log转换后的分布直方图
    plt.figure()
    sns.set_style('white')
    sns.distplot(series, hist=True, kde=True, fit=norm,
                 kde_kws={"label": "kernel density estimate"}, fit_kws={"label": "estimated normal PDF"})
    plt.legend(loc='upper right')
    sns.plt.title('%s Distribution(log transformed)' % name)
    sns.plt.xlabel(name)
    sns.plt.ylabel('Frequency')
    plt.savefig('%s-distplot-log.png' % name)
    # sns.plt.show()
    plt.close()
    plt.figure()
    stats.probplot(series, plot=plt)
    plt.savefig('%s-probability-plot-log.png' % name)
    # plt.show()
    plt.close()


def box(series1, series2):
    dat = pd.concat([series1, series2], axis=1)
    sns.set_style('white')
    sns.boxplot(x='month', y="pm", data=dat)
    sns.plt.title('Beijing PM2.5 Monthly Distribution')
    sns.plt.xlabel('Month')
    sns.plt.ylabel('PM2.5')
    # plt.xticks(rotation=90)
    plt.show()


def violin(series1, series2):
    dat = pd.concat([series1, series2], axis=1)
    sns.set_style('white')
    sns.violinplot(x='months', y="pm", data=dat,
                   order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
    # sns.plt.title('Beijing PM2.5 Monthly Distribution', fontsize=20)
    sns.plt.xlabel('Month', fontsize=20)
    sns.plt.ylabel('PM2.5 concentrations', fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    # plt.show()
    plt.savefig('E:/Pycharm/Mywork/aqi/log/EDA/PM25/violin-pm25-4.png', bbox_inches='tight', dpi=400)


def heat_map(frame):
    corrmat = frame.corr()
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix (heatmap style)')
    sns.plt.show()


def heat_map_n(frame):
    k = 15  # number of variables for heatmap , gist_heat
    corrmat = frame.corr()
    cols = corrmat.nlargest(k, 'pm2.5')['pm2.5'].index
    cm = np.corrcoef(frame[cols].values.T)
    # plt.subplots(figsize=(9, 9))
    # plt.figure()
    sns.set(font_scale=1.3)
    sns.heatmap(cm, cmap='jet', cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10}, square=True,
                yticklabels=cols.values, xticklabels=cols.values)  # cmap='gist_heat'
    # plt.title('PM2.5 Correlation Matrix (zoomed heatmap style)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.show()
    plt.savefig('E:/Pycharm/Mywork/aqi/log/EDA/PM25/heatmap4.png', bbox_inches='tight', dpi=300)


def cloud(frame, cols):
    sns.set(context='notebook', style='white')
    sns.pairplot(frame[cols], size=2.5, kind='scatter', hue='class-bj', plot_kws={'s': 30}, palette=sns.hls_palette(6, l=.3, s=.8),
                 vars=['pm2.5-bj', 'pm10-bj', 'co-bj', 'no2-bj', 'so2-bj', 'o3-bj'])
    # sns.pairplot(frame[cols], size=2.5, kind='scatter', hue='class-cd-1', plot_kws={'s': 30}, palette=sns.hls_palette(7, l=.3, s=.8))
    plt.savefig('E:/Pycharm/Mywork/aqi/log/EDA/PM25/cloud/cloud-12.png', bbox_inches='tight', dpi=400)
    # plt.show()


def scatter(series1, series2):
    sns.set(style='white')
    # plt.scatter(series1.values, series2.values, marker='o')
    dat = pd.concat([series1, series2], axis=1)
    sns.stripplot(x='WD-ext', y="pm2.5-bj", data=dat, hue='WD-ext', palette=sns.hls_palette(16, l=.3, s=.8))
    # plt.title('scatter plot for PM2.5 and wind direction', fontsize=20)
    plt.xlabel('Wind direction in category', fontsize=20)
    plt.ylabel('PM2.5 concentrations', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # plt.savefig('E:/Pycharm/Mywork/aqi/log/EDA/PM25/WD-32.png', bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    path = 'E:/Pycharm/Mywork/aqi/data/3.offshelf/aqi+54511-1.xlsx'
    data = pd.read_excel(path)
    # print(data.columns)
    # print(data['o3-bj'].describe())
    # print(data['o3-bj'].skew())
    # print(data['o3-bj'].kurt())

    # distribution(data['aqi-bj-1'], 'aqi-bj')
    # transformed = np.log1p(data['aqi-bj-1'])
    # print(transformed.skew())
    # print(transformed.kurt())
    # distribution(transformed, 'aqi-bj (Box-Cox transformed)')
    '''
    for item in data.columns:
        if item not in ['GST-mean', 'GST-max', 'GST-min', 'TEM-mean', 'TEM-max', 'TEM-min']:
            distribution(data[item], item)
            transformed = np.log1p(data[item])
            # transformed = np.log([i if i != 0 else 1 for i in data[item].values])
            distribution(transformed, item+' (log transformed)')
            print(item)
        else:
            continue
    '''
    # box(data['pm'], data['month'])
    # violin(data['pm'], data['month'])
    # violin(data['pm'], data['months'])

    # heat_map(data)
    heat_map_n(data)

    # column = ['target', 'pm2.5-zjk', 'pm10-zjk', 'so2-bj', 'o3-bj', 'EVP', 'WIN-ext', 'class-bj']
    # column = ['pm', 'aqi-bj-1', 'pm2.5-bj-1', 'pm10-bj-1', 'co-bj-1', 'no2-bj-1', 'so2-bj-1', 'o3-bj-1', 'class-bj-1']
    # column = ['pm', 'aqi-lf-1', 'pm2.5-lf-1', 'pm10-lf-1', 'co-lf-1', 'no2-lf-1', 'so2-lf-1', 'o3-lf-1', 'class-lf-1']
    # column = ['pm', 'aqi-ts-1', 'pm2.5-ts-1', 'pm10-ts-1', 'co-ts-1', 'no2-ts-1', 'so2-ts-1', 'o3-ts-1', 'class-ts-1']
    # column = ['pm', 'aqi-tj-1', 'pm2.5-tj-1', 'pm10-tj-1', 'co-tj-1', 'no2-tj-1', 'so2-tj-1', 'o3-tj-1', 'class-tj-1']
    # column = ['pm', 'aqi-zjk-1', 'pm2.5-zjk-1', 'pm10-zjk-1', 'co-zjk-1', 'no2-zjk-1', 'so2-zjk-1', 'o3-zjk-1', 'class-zjk-1']
    # column = ['pm', 'aqi-cd-1', 'pm2.5-cd-1', 'pm10-cd-1', 'co-cd-1', 'no2-cd-1', 'so2-cd-1', 'o3-cd-1', 'class-cd-1']
    # column = ['pm', 'aqi-bd-1', 'pm2.5-bd-1', 'pm10-bd-1', 'co-bd-1', 'no2-bd-1', 'so2-bd-1', 'o3-bd-1', 'class-bd-1']
    # column = ['pm', 'GST-mean', 'GST-max', 'GST-min', 'TEM-mean', 'TEM-max', 'TEM-min', 'SSD', 'class-bj-1']
    # column = ['pm', 'PRE-208', 'PRE-820', 'PRE-2020', 'EVP', 'RHU-mean', 'RHU-min', 'class-bj-1']
    # column = ['pm', 'WIN-mean', 'WIN-max', 'WD-max', 'WIN-ext', 'WD-ext', 'PRS-mean', 'PRS-max', 'PRS-min', 'class-bj-1']
    # column = ['pm2.5-bj', 'pm10-bj', 'co-bj', 'no2-bj', 'so2-bj', 'o3-bj', 'class-bj']
    # cloud(data, column)

    # scatter(data['WD-ext'], data['pm2.5-bj'])
