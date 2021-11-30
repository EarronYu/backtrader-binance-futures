# -*- coding: utf-8 -*- 
'''
TopQuant-TQ极宽智能量化回溯分析系统2019版

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2019.01.011 首发

网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化总群，124134140
      
  
  
文件名:toolkit.py
默认缩写：import topquant2019 as tk
简介：Top极宽量化·常用量化系统参数模块
 

'''
#

import sys, os, re
import arrow, bs4, random
import numexpr as ne
#

# import reduce #py2
from functools import reduce  # py3
import itertools
import collections
#
# import cpuinfo as cpu
import psutil as psu
from functools import wraps
import datetime as dt
import copy
#
import numpy as np
import pandas as pd
import tushare as ts
# import talib as ta


import matplotlib as mpl
import matplotlib.colors
from matplotlib import cm
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
# import multiprocessing
#
import pyfolio as pf
from pyfolio.utils import (to_utc, to_series)
#
import backtrader as bt
import backtrader.observers as btobv
import backtrader.indicators as btind
import backtrader.analyzers as btanz
import backtrader.feeds as btfeeds
#
from backtrader.analyzers import SQN, AnnualReturn, TimeReturn, SharpeRatio, TradeAnalyzer
#
import topq_talib as tqta
#
from io import BytesIO
import base64

#


# -------------------
# ----glbal var,const
__version__ = '2019.M1'

sgnSP4 = '    '
sgnSP8 = sgnSP4 + sgnSP4
#
corlst = ['#0000ff', '#000000', '#00ff00', '#0000FF', '#8A2BE2', '#A52A2A', '#5F9EA0', '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00FFFF', '#00008B',
          '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
          '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF',
          '#FFD700', '#DAA520', '#808080', '#008000', '#ADFF2F', '#F0FFF0', '#FF69B4', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5',
          '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA', '#778899',
          '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE',
          '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000', '#6B8E23', '#FFA500',
          '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6', '#800080',
          '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#FAA460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090',
          '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3', '#FFFFFF', '#F5F5F5', '#FFFF00',
          '#9ACD32']

# @ datasires.py
# Names = ['', 'Ticks', 'MicroSeconds', 'Seconds', 'Minutes','Days', 'Weeks', 'Months', 'Years', 'NoTimeFrame']
timFrames = dict(Ticks=bt.TimeFrame.Ticks, MicroSeconds=bt.TimeFrame.MicroSeconds, Seconds=bt.TimeFrame.Seconds, Minutes=bt.TimeFrame.Minutes
                 , Days=bt.TimeFrame.Days, Weeks=bt.TimeFrame.Weeks, Months=bt.TimeFrame.Months, Years=bt.TimeFrame.Years, NoTimeFrame=bt.TimeFrame.NoTimeFrame)

#
rdat0 = '/TQDat/'
rdatDay = rdat0 + "day/"
rdatDayInx = rdatDay + "inx/"
rdatDayEtf = rdatDay + "etf/"
#
rdatMin0 = rdat0 + "min/"
rdatTick0 = rdat0 + "tick/"
rdatReal0 = rdat0 + "real/"

#
ohlcLst = ['open', 'high', 'low', 'close']
ohlcVLst = ohlcLst + ['volume']
#
ohlcDLst = ['date'] + ohlcLst
ohlcDVLst = ['date'] + ohlcVLst
#
ohlcDExtLst = ohlcDVLst + ['adj close']
ohlcBTLst = ohlcDVLst + ['openinterest']  # backtrader
#
# ----kline
tq10_corUp, tq10_corDown = ['#7F7F7F', '#17BECF']  # plotly
tq09_corUp, tq09_corDown = ['#B61000', '#0061B3']
tq08_corUp, tq08_corDown = ['#FB3320', '#020AF0']
tq07_corUp, tq07_corDown = ['#B0F76D', '#E1440F']
tq06_corUp, tq06_corDown = ['#FF3333', '#47D8D8']
tq05_corUp, tq05_corDown = ['#FB0200', '#007E00']
tq04_corUp, tq04_corDown = ['#18DEF5', '#E38323']
tq03_corUp, tq03_corDown = ['black', 'blue']
tq02_corUp, tq02_corDown = ['red', 'blue']
tq01_corUp, tq01_corDown = ['red', 'lime']
#
tq_ksty01 = dict(volup=tq01_corUp, voldown=tq01_corDown, barup=tq01_corUp, bardown=tq01_corDown)
tq_ksty02 = dict(volup=tq02_corUp, voldown=tq02_corDown, barup=tq02_corUp, bardown=tq02_corDown)
tq_ksty03 = dict(volup=tq03_corUp, voldown=tq03_corDown, barup=tq03_corUp, bardown=tq03_corDown)
tq_ksty04 = dict(volup=tq04_corUp, voldown=tq04_corDown, barup=tq04_corUp, bardown=tq04_corDown)
tq_ksty05 = dict(volup=tq05_corUp, voldown=tq05_corDown, barup=tq05_corUp, bardown=tq05_corDown)
tq_ksty06 = dict(volup=tq06_corUp, voldown=tq06_corDown, barup=tq06_corUp, bardown=tq06_corDown)
tq_ksty07 = dict(volup=tq07_corUp, voldown=tq07_corDown, barup=tq07_corUp, bardown=tq07_corDown)
tq_ksty08 = dict(volup=tq08_corUp, voldown=tq08_corDown, barup=tq08_corUp, bardown=tq08_corDown)
tq_ksty09 = dict(volup=tq09_corUp, voldown=tq09_corDown, barup=tq09_corUp, bardown=tq09_corDown)
tq_ksty10 = dict(volup=tq10_corUp, voldown=tq10_corDown, barup=tq10_corUp, bardown=tq10_corDown)


# -------------------


# --------------------


class TQ_bar(object):
    ''' 
    设置TopQuant项目的各个全局参数
    尽量做到all in one

    '''

    def __init__(self):
        # ----rss.dir
        #
        # BT回测核心变量Cerebro,缩:：cb
        self.cb = None
        #
        # BT回测默认参数
        self.prjNm = ''  # 项目名称
        self.cash0 = 100000  # 启动最近 10w
        self.trd_mod = 1  # 交易模式：1，定量交易(默认)；2，现金额比例交易
        self.stake0 = 100  # 定量交易，每次交易数目，默认为 100 手
        self.ktrd0 = 30  # 比例交易,每次交易比例，默认为 30%

        # 数据目录
        self.rdat0 = ''  # 产品(股票/基金/期货等)数据目录
        self.rbas0 = ''  # 对比基数(指数等)数据目录
        #
        self.pools = {}  # 产品(股票/基金/期货等)池，dict字典格式
        self.pools_code = {}  # 产品代码(股票/基金/期货等)池，dict字典格式
        #
        # ------bt.var
        # 分析模式： 0，base基础分析; 1, 交易底层数据分析
        # pyfolio专业图表分析，另外单独调用
        self.anz_mod = 1
        self.bt_results = None  # BT回测运行结果数据，主要用于分析模块
        #
        self.tim0, self.tim9 = None, None  # BT回测分析起始时间、终止时间
        self.tim0str, self.tim9str = '', ''  # BT回测分析起始时间、终止时间，字符串格式
        #


# ----------------------

# ----------top.quant.2019

def tq_init(prjNam='TQ01', cash0=100000.0, stake0=100):
    #
    def _xfloat3(x):
        return '%.3f' % x

    # ----------
    #
    # 初始化系统环境参数,设置绘图&数据输出格式
    mpl.style.use('seaborn-whitegrid');
    pd.set_option('display.width', 450)
    # pd.set_option('display.float_format', lambda x: '%.3g' % x)
    pd.set_option('display.float_format', _xfloat3)
    np.set_printoptions(suppress=True)  # 取消科学计数法 #as_num(1.2e-4)
    #
    #
    # 设置部分BT量化回测默认参数，清空全局股票池、代码池
    qx = TQ_bar()
    qx.prjName, qx.cash0, qx.stake0 = prjNam, cash0, stake0
    qx.pools, qx.pools_code = {}, {}
    #
    #
    return qx


# ----------bt.xxx


def plttohtml(plt, filename):
    # plt.show()

    # 转base64
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())  # 将图片转为base64
    figdata_str = str(figdata_png, "utf-8")  # 提取base64的字符串，不然是b'xxx'

    # 保存为.html
    html = '<img src=\"data:image/png;base64,{}\"/>'.format(figdata_str)
    if filename is None:
        filename = 'result' + '.html'
    with open(filename + '.html', 'w') as f:
        f.write(html)


def bt_set(qx, anzMod=0):
    # 设置BT回测变量Cerebro
    # 设置简化名称 
    # 初始化回测数据池,重新导入回测数据
    # 设置各种BT回测初始参数
    # 设置分析参数
    #
    # 设置BT回测核心变量Cerebro
    qx.cb = bt.Cerebro()
    #
    # 设置简化名称
    qx.anz, qx.br = bt.analyzers, qx.cb.broker
    # bt:backtrader,ind:indicators,p:param

    #
    # 初始化回测数据池,重新导入回测数据
    pools_2btdata(qx)

    #
    # 设置各种BT回测初始参数
    qx.br.setcash(qx.cash0)
    qx.br.setcommission(commission=0.001)
    qx.br.set_slippage_fixed(0.01)
    #
    # 设置交易默认参数
    qx.trd_mod = 1
    qx.ktrd0 = 30
    qx.cb.addsizer(bt.sizers.FixedSize, stake=qx.stake0)
    #
    #
    # 设置分析参数
    qx.cb.addanalyzer(qx.anz.Returns, _name="Returns")
    qx.cb.addanalyzer(qx.anz.DrawDown, _name='DW')
    # SharpeRatio夏普指数
    qx.cb.addanalyzer(qx.anz.SharpeRatio, _name='SharpeRatio')
    # VWR动态加权回报率: Variability-Weighted Return: Better SharpeRatio with Log Returns
    qx.cb.addanalyzer(qx.anz.VWR, _name='VWR')
    qx.cb.addanalyzer(SQN)
    #
    qx.cb.addanalyzer(qx.anz.AnnualReturn, _name='AnnualReturn')  # 年化回报率

    # 设置分析级别参数
    qx.anz_mod = anzMod
    if anzMod > 0:
        qx.cb.addanalyzer(qx.anz.TradeAnalyzer, _name='TradeAnalyzer')
        # cerebro.addanalyzer(TimeReturn, timeframe=timFrames['years'])
        # cerebro.addanalyzer(SharpeRatio, timeframe=timFrames['years'])
        #
    #
    qx.cb.addanalyzer(qx.anz.PyFolio, _name='pyfolio')

    #
    return qx


def bt_anz(qx):
    # 分析BT量化回测数据
    print('\nanz...')
    #
    dcash0, dval9 = qx.br.startingcash, qx.br.getvalue()
    dget = dval9 - dcash0
    # kret=dval9/dcash0*100
    kget = dget / dcash0 * 100
    #
    strat = qx.bt_results[0]
    anzs = strat.analyzers
    #
    #
    # dsharp=anzs.SharpeRatio.get_analysis()['sharperatio']
    dsharp = anzs.SharpeRatio.get_analysis()['sharperatio']
    if dsharp == None: dsharp = 0
    #
    if qx.anz_mod > 1:
        trade_info = anzs.TradeAnalyzer.get_analysis()
    #
    dw = anzs.DW.get_analysis()
    max_drowdown_len = dw['max']['len']
    max_drowdown = dw['max']['drawdown']
    max_drowdown_money = dw['max']['moneydown']
    # --------
    print('\n-----------anz lv# 1 ----------')
    print('\nBT回测数据分析')
    print('时间周期：%s 至 %s' % (qx.tim0str, qx.tim9str))
    # print('%s终止时间：%s'% (sgnSP4,qx.tim9str))
    print('==================================================')
    print('起始资金 Starting Portfolio Value: %.2f' % dcash0)
    print('资产总值 Final Portfolio Value: %.2f' % dval9)
    print('利润总额 Total Profit:  %.2f' % dget)
    print('ROI投资回报率 Return on Investment: %.2f %%' % kget)

    print('==================================================')
    #
    print('夏普指数 SharpeRatio : %.2f' % dsharp)
    print('最大回撤周期 max_drowdown_len : %.2f' % max_drowdown_len)
    print('最大回撤 max_drowdown : %.2f' % max_drowdown)
    print('最大回撤(资金) max_drowdown_money : %.2f' % max_drowdown_money)
    print('==================================================\n')
    #
    if qx.anz_mod > 1:
        print('\n-----------anz lv# %d ----------\n' % qx.anz_mod)

        for dat in anzs:
            dat.print()


def bt_anz_folio(qx):
    # 分析BT量化回测数据
    # 专业pyFolio量化分析图表
    #
    print('\n-----------pyFolio----------')
    strat = qx.bt_results[0]
    anzs = strat.analyzers
    #
    xpyf = anzs.getbyname('pyfolio')
    xret, xpos, xtran, gross_lev = xpyf.get_pf_items()
    #
    # xret.to_csv('tmp/x_ret.csv',index=True,header=None,encoding='utf8')
    # xpos.to_csv('tmp/x_pos.csv',index=True,encoding='utf8')
    # xtran.to_csv('tmp/x_tran.csv',index=True,encoding='utf8')
    #
    xret, xpos, xtran = to_utc(xret), to_utc(xpos), to_utc(xtran)
    #
    # 创建瀑布(活页)式分析图表
    # 部分图表需要联网现在spy标普数据，
    # 可能会出现"假死"现象，需要人工中断
    pf.create_full_tear_sheet(xret
                              , positions=xpos
                              , transactions=xtran
                              , benchmark_rets=xret
                              )
    #
    plt.show()


'''
【ps，附录：专业pyFolio量化分析图表图片函数接口API】
有关接口函数API，不同版本差异很大，请大家注意相关细节
def create_full_tear_sheet(returns,
                           positions=None,
                           transactions=None,
                           market_data=None,
                           benchmark_rets=None,
                           slippage=None,
                           live_start_date=None,
                           sector_mappings=None,
                           bayesian=False,
                           round_trips=False,
                           estimate_intraday='infer',
                           hide_positions=False,
                           cone_std=(1.0, 1.5, 2.0),
                           bootstrap=False,
                           unadjusted_returns=None,
                           set_context=True):
    
pf.create_full_tear_sheet(
#pf.create_returns_tear_sheet(
        test_returns
        ,positions=test_pos
        ,transactions=test_txn
        ,benchmark_rets=test_returns
        #, live_start_date='2004-01-09'
        )
'''


# ----------pools.data.xxx
def pools_get4fn(fnam, tim0str, tim9str, fgSort=True, fgCov=True):
    '''
     从csv文件，数据读取函数，兼容csv标准OHLC数据格式文件
    
    【输入参数】 
       fnam：csv数据文件名
       tim0str,tim9str：回测起始时间，终止时间，字符串格式
       fgSort：正序排序标志，默认为 True
    【输出数据】 
       data：BT回测内部格式的数据包
    '''

    # skiprows=skiprows,header=header,parse_dates=True, index_col=0,
    # df = pd.read_hdf(fnam, index_col=1, parse_dates=True, key='df', mode='r')
    # df = pd.DataFrame(df)
    # df.set_index('candle_begin_time', inplace=True)
    # print(df)
    df = pd.read_csv(fnam, index_col=0, parse_dates=True)
    df.sort_index(ascending=fgSort, inplace=True)  # True：正序
    df.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M:%S.%fZ')
    #
    tim0 = None if tim0str == '' else dt.datetime.strptime(tim0str, '%Y-%m-%d')
    tim9 = None if tim9str == '' else dt.datetime.strptime(tim9str, '%Y-%m-%d')
    # prDF(df)
    # xxx
    #
    df['openinterest'] = 0
    if fgCov:
        data = bt.feeds.PandasData(dataname=df, fromdate=tim0, todate=tim9)
    else:
        data = df
    #
    return data


def pools_get4df(df, tim0str, tim9str, fgSort=True, fgCov=True):
    '''
     从csv文件，数据读取函数，兼容csv标准OHLC数据格式文件

    【输入参数】
       fnam：csv数据文件名
       tim0str,tim9str：回测起始时间，终止时间，字符串格式
       fgSort：正序排序标志，默认为 True
    【输出数据】
       data：BT回测内部格式的数据包
    '''

    # skiprows=skiprows,header=header,parse_dates=True, index_col=0,
    # df = pd.read_hdf(fnam, index_col=1, parse_dates=True, key='df', mode='r')
    # df = pd.DataFrame(df)
    # df.set_index('candle_begin_time', inplace=True)
    # print(df)
    # prDF(df)
    # xxx
    #

    if fgCov:
        df['openinterest'] = 0
        df.sort_index(ascending=fgSort, inplace=True)  # True：正序
        df.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M:%S')
        #
        tim0 = None if tim0str == '' else dt.datetime.strptime(tim0str, '%Y-%m-%d')
        tim9 = None if tim9str == '' else dt.datetime.strptime(tim9str, '%Y-%m-%d')

        data = bt.feeds.PandasData(dataname=df, fromdate=tim0, todate=tim9)
    else:
        # Create a Data Feed
        tim0 = None if tim0str == '' else dt.datetime.strptime(tim0str, '%Y-%m-%d')
        tim9 = None if tim9str == '' else dt.datetime.strptime(tim9str, '%Y-%m-%d')
        data = bt.feeds.GenericCSVData(
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
            dataname=df,
            fromdate=tim0,
            todate=tim9,
            nullvalue=0.0,
            dtformat=('%Y-%m-%d %H:%M:%S'),
            tmformat=('%H:%M:%S'),
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            reverse=False)
    #
    # print(data)
    # data.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M:%S.%fZ')
    return data


def pools_get4flst(qx, rdat0, syblst, tim0str='', tim9str='', fgInx=False, fgPr=False, fgCov=True):
    '''
     从csv文件，批量读取数据，并转换文件BT回测内部格式，加入pools产品池
    
    【输入参数】 
       qx：全局变量
       rdat0：数据文件目录
       syblst：产品代码列表
       tim0str,tim9str：回测起始时间，终止时间，字符串格式
       fgInx： 指数数据模式，默认为 False，产品(股票数据)
    【输出数据】：
       输出数据全部保存在全局变量qx当中：
       qx.tim0,qx.tim9：回测起始时间，终止时间，datetime格式
       qx.tim0str,qx.tim9str：回测起始时间，终止时间，字符串格式
       qx.pools: 产品(股票/基金/期货等)池数据，dict字典格式
       qx.pools_code: 产品(股票/基金/期货等)代码池数据，dict字典格式
       qx.rdat0: 产品(股票/基金/期货等)数据目录
       qx.rbas0: 对比基数(指数等)数据目录
    
    '''
    qx.tim0str, qx.tim9str = tim0str, tim9str
    qx.tim0 = None if tim0str == '' else dt.datetime.strptime(tim0str, '%Y-%m-%d')
    qx.tim9 = None if tim9str == '' else dt.datetime.strptime(tim9str, '%Y-%m-%d')
    #
    if fgPr:
        # print('\n')
        prlog(['@pools_get4flst', rdat0, syblst])

    #
    for xc, cod in enumerate(syblst):
        fss, css = rdat0 + cod + '.csv', cod
        if fgInx: css = 'inx_' + css
        #
        # if fgPr:
        #    prlog(['@pools_get4flst',xc,'#',fgInx,fss,css])
        dat = pools_get4fn(fss, tim0str, tim9str, fgSort=True, fgCov=fgCov)
        # qx.cb.adddata(dat,name=css)
        #
        qx.pools_code[css] = [cod, css, fss]
        qx.pools[css] = dat

    #
    if fgInx:
        qx.rbas0 = rdat0
    else:
        qx.rdat0 = rdat0

    # print('\nqx.sybPools:',qx.sgnPools)


def pools_2btdata(qx):
    for xcod, vlst in qx.pools_code.items():
        cod, css, fss = vlst
        # print('@',xcod,vlst,'#',cod,css,fss)
        dat = qx.pools[css]
        qx.cb.adddata(dat, name=css)


# -----------tools。xxx

# ------lst,dict,....
def lst_2typ(dlst, xtyp=float):
    vlst = list(map(xtyp, dlst))
    #
    return vlst


def lst_xmul(lists, xdiv=','):
    '''输入多个列表组成的列表, 输出其中每个列表所有元素可能的所有排列组合
    xdiv 用于分隔每个元素'''

    def _xmul(list1, list2):
        return [str(x) + xdiv + str(y) for x in list1 for y in list2]

    #
    return reduce(_xmul, lists)


# -----------pr.xxx
def prlog(xlst):
    print('\t', xlst)


def prx(x, dss=''):
    print('\n', dss)
    print(x)


def prDF(df0, dss='', n=10, nfloat=2):
    if df0 is None: return
    #
    df = df0.round(nfloat)
    #
    print('\n', dss, len(df.index))
    if len(df.index) > 0:
        print(df.head(n))
    if len(df.index) > n:
        print('')
        print(df.tail(n))
    #
    print('\nlen-DF:', len(df.index))


def prObj(rx, css=''):
    print('\n', css)
    for xnam, xdat in vars(rx).items():
        dss = '{0:15} = '.format(xnam)
        dat = '{0}'.format(xdat)
        if len(dat) > 50: dat = '......'
        dss = ''.join([dss, dat])
        print(dss)
        # t5.append(dss)
    # sorted(t5)
    # t5.sort()
    # prLst(t5)


# -----misc

def sort_dict(xdict, fgRev=False):
    # sorted(d.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    dat = dict(sorted(xdict.items(), key=lambda k: k[1], reverse=fgRev))

    #
    return dat


def timNSec(tim, tim0, fgPr=False):
    if tim == '': tim = arrow.now()
    if type(tim) == str: tim = arrow.get(tim)
    if type(tim0) == str: tim0 = arrow.get(tim0)
    tn = tim - tim0
    xn = round(tn.total_seconds(), 2)
    if fgPr: print(xn, 's,', tim.format('HH:mm:ss'), ',t0,', tim0.format('HH:mm:ss'))
    #    
    return xn
