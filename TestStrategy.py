import os

import quantstats
import toolkit as tk
from PrepareCSV import prepare_data
import backtrader as bt
from backtrader.analyzers import SQN, AnnualReturn, TimeReturn, SharpeRatio,TradeAnalyzer
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
import warnings
# import strategy
from strategies.BBReverse import BBReverse
from strategies.SimpleBollinger import SimpleBollinger
from strategies.BBKCReverse import BBKCReverse
from strategies.BBKCBreak import BBKCBreak


class CommInfoFractional(bt.CommissionInfo):
    def getsize(self, price, cash):
        '''Returns fractional size for cash operation @price'''
        return self.p.leverage * (cash / price)


class FixedReverser(bt.Sizer):
    params = (('stake', 50),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        size = self.p.stake * (1 + (position.size != 0))
        return size


class MyBuySell(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker='^', markersize=4.0, color='green', fillstyle='full'),
        sell=dict(marker='v', markersize=4.0, color='red', fillstyle='full')
        #
        # buy=dict(marker='$++$', markersize=12.0),
        # sell=dict(marker='$--$', markersize=12.0)
        #
        # buy=dict(marker='$✔$', markersize=12.0),
        # sell=dict(marker='$✘$', markersize=12.0)
    )


class MyTrades(bt.observers.Trades):
    plotlines = dict(
        pnlplus=dict(_name='Positive',
                     ls='', marker='o', color='blue',
                     markersize=4.0, fillstyle='full'),
        pnlminus=dict(_name='Negative',
                      ls='', marker='o', color='red',
                      markersize=4.0, fillstyle='full')
    )


def runstrat(args=None):
    # setup entrance
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BBKCBreak)
    # cerebro.addstrategy(SimpleBollinger)
    # prepare data
    t0str, t9str = '2019-01-01', '2021-10-01'
    symbol = 'ETHUSDT'
    fgCov = False
    df = prepare_data(t0str, t9str, symbol, fgCov=fgCov, prep_new=True, mode='test')
    data = tk.pools_get4df(df, t0str, t9str, fgCov=fgCov)
    dataha = data.clone()
    dataha.addfilter(bt.filters.HeikinAshi(dataha))
    tframes = dict(minutes=bt.TimeFrame.Minutes, days=bt.TimeFrame.Days, weeks=bt.TimeFrame.Weeks, months=bt.TimeFrame.Months, years=bt.TimeFrame.Years)
    cerebro.resampledata(data, timeframe=tframes['minutes'], compression=5, name='Real')
    cerebro.resampledata(dataha, timeframe=tframes['minutes'], compression=5, name='Heikin')
    # cerebro.adddata(data)
    dmoney0 = 100.0
    cerebro.broker.setcash(dmoney0)
    dcash0 = cerebro.broker.startingcash
    commission = 0.0004
    comminfo = CommInfoFractional(commission=commission)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.addsizer(bt.sizers.FixedSize)
    bt.observers.BuySell = MyBuySell
    bt.observers.Trades = MyTrades
    # 设置pyfolio分析参数
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    #
    print('\n\t#运行cerebro')
    results = cerebro.run(maxcpus=True)
    # results = cerebro.run(runonce=False, exactbars=-2)
    print('\n#基本BT量化分析数据')
    dval9 = cerebro.broker.getvalue()
    dget = dval9 - dcash0
    kret = dget / dcash0 * 100
    # 最终投资组合价值
    strat = results[0]
    print('\t起始资金 Starting Portfolio Value: %.2f' % dcash0)
    print('\t资产总值 Final Portfolio Value: %.2f' % dval9)
    print('\t利润总额:  %.2f,' % dget)
    print('\tROI投资回报率 Return on investment: %.2f %%' % kret)

    print('\n==================================================')
    print('\n quantstats专业量化分析图表\n')
    warnings.filterwarnings('ignore')
    portfolio_stats = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)
    quantstats.reports.html(returns, output='..//data//teststats.html', title='Crypto Sentiment')
    #
    print('\n#绘制BT量化分析图形')
    # try:
    #     b = Bokeh(plot_mode='single', output_mode='show', filename='..//data//report.html')
    #     cerebro.plot(b)
    # except:
    cerebro.plot(style='candlestick')


if __name__ == '__main__':
    runstrat()
