import warnings
from datetime import datetime
import optunity.metrics
import toolkit as tk
from PrepareCSV import prepare_data
from backtrader_plotting import Bokeh
import quantstats
import math
import backtrader as bt
from strategies.BBKCReverse import BBKCReverse


class CommInfoFractional(bt.CommissionInfo):
    def getsize(self, price, cash):
        '''Returns fractional size for cash operation @price'''
        return self.p.leverage * (cash / price)


class AllSizer(bt.Sizer):
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy is True:
            return math.floor(cash / data.high)
        if isbuy is False:
            return math.floor(cash / data.low)
        else:
            return self.broker.getposition(data)


class MyBuySell(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker='^', markersize=4.0, color='lime', fillstyle='full'),
        sell=dict(marker='v', markersize=4.0, color='red', fillstyle='full')
        #
        # buy=dict(marker='$++$', markersize=12.0),
        # sell=dict(marker='$--$', markersize=12.0)
        #
        # buy=dict(marker='$✔$', markersize=12.0),
        # sell=dict(marker='$✘$', markersize=12.0)
    )


# ----------------------
# prepare data
t0str, t9str = '2019-01-01', '2020-01-01'
symbol = 'ETHUSDT'
fgCov = False
df = prepare_data(t0str, t9str, symbol, fgCov=fgCov, prep_new=True, mode='opt')
# print(df)
data = tk.pools_get4df(df, t0str, t9str, fgCov=fgCov)
dataha = data.clone()
dataha.addfilter(bt.filters.HeikinAshi(dataha))


def runstrat(window, bbdevs, kcdevs, bias_pct, volatile_pct):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BBKCReverse, window=int(window), bbdevs=bbdevs, bias_pct=bias_pct, kcdevs=int(kcdevs), volatile_pct=volatile_pct)
    dmoney0 = 100.0
    cerebro.broker.setcash(dmoney0)
    # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio', timeframe=bt.TimeFrame.Months, compression=3, factor=4)
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    commission = 0.0004
    comminfo = CommInfoFractional(commission=commission)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.addsizer(bt.sizers.FixedSize)
    tframes = dict(minutes=bt.TimeFrame.Minutes, days=bt.TimeFrame.Days, weeks=bt.TimeFrame.Weeks, months=bt.TimeFrame.Months, years=bt.TimeFrame.Years)
    cerebro.resampledata(data, timeframe=tframes['minutes'], compression=1, name='Real')
    cerebro.resampledata(dataha, timeframe=tframes['minutes'], compression=1, name='Heikin')
    # cerebro.adddata(data)
    results = cerebro.run()
    strat = results[0]
    anzs = strat.analyzers
    #
    warnings.filterwarnings('ignore')
    portfolio_stats = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)
    dsharp = quantstats.stats.smart_sharpe(returns, periods=365)
    sortino = quantstats.stats.smart_sortino(returns, periods=365)
    sqn = anzs.sqn.get_analysis().sqn
    dcash9 = cerebro.broker.getvalue()
    dcash0 = cerebro.broker.startingcash
    pnl = dcash9 / dcash0 - 1
    if dsharp is not None:
        if (dsharp > 0) and (sqn > 1) and (sortino > 0) and (pnl > 0):
            param = sqn * dsharp * pnl**0.5 * sortino
        else:
            param = 0
    else:
        param = 0
    # return
    print(f'(分析指标 SQN={sqn}, Sharp_Ratio={dsharp}, Sortino={sortino}, pnl={pnl})')
    return param


starttime = datetime.now()
opt = optunity.maximize(runstrat, num_evals=100, solver_name='particle swarm', window=[1, 200], bbdevs=[0.01, 5.0], kcdevs=[0.01, 5.0], bias_pct=[0.01, 0.20], volatile_pct=[0.01, 0.20])
# long running
endtime = datetime.now()
duringtime = endtime - starttime
print('time cost: ', duringtime)
# 得到最优参数结果
optimal_pars, details, _ = opt
print('Optimal Parameters:')
print('策略参数 window=%s, bbdevs=%s, kcdevs=%s, bias_pct=%s, volatile_pct=%s' % (optimal_pars['window'], optimal_pars['bbdevs'], optimal_pars['kcdevs'], optimal_pars['bias_pct'], optimal_pars['volatile_pct']))
# 利用最优参数最后回测一次，作图
cerebro = bt.Cerebro()
cerebro.addstrategy(BBKCReverse, window=int(optimal_pars['window']), bbdevs=optimal_pars['bbdevs'], bias_pct=optimal_pars['bias_pct'], kcdevs=optimal_pars['kcdevs'], volatile_pct=optimal_pars['volatile_pct'])

dmoney0 = 100.0
cerebro.broker.setcash(dmoney0)
dcash0 = cerebro.broker.startingcash
commission = 0.0004
comminfo = CommInfoFractional(commission=commission)
cerebro.broker.addcommissioninfo(comminfo)
cerebro.addsizer(bt.sizers.FixedSize)
tframes = dict(minutes=bt.TimeFrame.Minutes, days=bt.TimeFrame.Days, weeks=bt.TimeFrame.Weeks, months=bt.TimeFrame.Months, years=bt.TimeFrame.Years)
cerebro.resampledata(data, timeframe=tframes['minutes'], compression=5, name='Real')
cerebro.resampledata(dataha, timeframe=tframes['minutes'], compression=5, name='Heikin')
bt.observers.BuySell = MyBuySell

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
quantstats.reports.html(returns, output='..//data//optstats.html', title='Crypto Sentiment')
#
print('\n#绘制BT量化分析图形')
# try:
#     b = Bokeh(plot_mode='single', output_mode='save', filename='..//data//optreport.html')
#     cerebro.plot(b)
# except:
cerebro.plot()

#
#
