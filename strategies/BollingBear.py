import backtrader as bt
import functions.toolkit as tk
from datetime import datetime
import arrow


class BollingerBear(bt.Strategy):
    """
    :param df:
    :param para: n, m,mean
    :return:
    # 布林线策略
    # 布林线中轨：n天收盘价的移动平均线
    # 布林线上轨：n天收盘价的移动平均线 + m * n天收盘价的标准差
    # 布林线上轨：n天收盘价的移动平均线 - m * n天收盘价的标准差
    # 当收盘价由下向上穿过上轨的时候，做多；然后由上向下穿过中轨的时候，平仓。
    # 当收盘价由上向下穿过下轨的时候，做空；然后由下向上穿过中轨的时候，平仓。
    根据Ryan萨博的思路,计算开仓价格与中线的比值
            https://bbs.quantclass.cn/thread/4521
    根据JIN的上下轨道使用平均差思路
            https://bbs.quantclass.cn/thread/4443
    """
    params = dict(bollwindow=200, devfactor=2, bias_pct=0.05)

    def log(self, txt, dt=None, fgPrint=False):
        # 增强型log记录函数，带fgPrint打印开关变量
        if fgPrint:
            dt = dt or self.datas[0].datetime.date(0)
            tn = tk.timNSec('', self.tim0wrk)
            # print('%s, %s，tn：%.2f' % (dt.isoformat(), txt))
            print('%s, %s，tim：%.2f s' % (dt.isoformat(), txt, tn))

    def __init__(self):
        self.bar_executed = len(self)
        self.buyprice = None
        self.buycomm = None
        self.sellprice = None
        self.sellcomm = None
        self.tim0wrk = arrow.now()
        self.dataclose = self.datas[0].close
        self.order = None

        midband = bt.indicators.MovingAverageSimple(self.data, period=self.params.bollwindow, plot=False)
        std = bt.indicators.StandardDeviation(self.data, period=self.params.bollwindow, plot=False)
        topband = midband + self.params.devfactor * std
        botband = midband - self.params.devfactor * std
        volatile = abs(self.data.open / midband - 1)
        low_volatile = bt.If(volatile < 0.05, 1, 0)
        bias = abs(self.data.close / midband - 1)
        low_bias = bt.If(bias <= self.params.bias_pct, 1, 0)
        crossuptop = bt.indicators.CrossUp(self.data.close, topband, plot=False)
        self.close_long = bt.indicators.CrossDown(self.data.close, midband, plot=False)
        crossdownbot = bt.indicators.CrossDown(self.data.close, botband, plot=False)
        self.close_short = bt.indicators.CrossUp(self.data.close, midband, plot=False)
        self.open_long = bt.And(crossuptop, low_volatile, low_bias)
        self.open_short = bt.And(crossdownbot, low_volatile, low_bias)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 检查订单执行状态order.status：
            # Buy/Sell order submitted/accepted to/by broker
            # broker经纪人：submitted提交/accepted接受,Buy买单/Sell卖单
            # 正常流程，无需额外操作
            return

        # 检查订单order是否完成
        # 注意: 如果现金不足，经纪人broker会拒绝订单reject order
        # 可以修改相关参数，调整进行空头交易
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('买单执行BUY EXECUTED,成交价： %.2f,小计 Cost: %.2f,佣金 Comm %.2f'
                         % (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log('卖单执行SELL EXECUTED,成交价： %.2f,小计 Cost: %.2f,佣金 Comm %.2f'
                         % (order.executed.price, order.executed.value, order.executed.comm))
                self.sellprice = order.executed.price
                self.sellcomm = order.executed.comm
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单Order： 取消Canceled/保证金Margin/拒绝Rejected')

        # 检查完成，没有交易中订单（pending order）
        self.order = None

    def notify_trade(self, trade):
        # 检查交易trade是关闭
        if not trade.isclosed:
            return

        self.log('交易操盘利润OPERATION PROFIT, 毛利GROSS %.2f, 净利NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        if self.order:
            return
        if self.open_long and not self.position:
            self.log('设置多单 BUY CREATE, @ %.2f' % self.dataclose[0])
            self.order = self.order_target_percent(target=0.99)
        if self.close_long and self.position:
            self.log('设置平多单 CLOSE LONG CREATE, @ %.2f' % self.dataclose[0])
            self.order = self.order_target_percent(target=0.00)
        if self.open_short and not self.position:
            self.log('设置空单 BUY CREATE, @ %.2f' % self.dataclose[0])
            self.order = self.order_target_percent(target=-0.99)
        if self.close_short and self.position:
            self.log('设置平空单 CLOSE LONG CREATE, @ %.2f' % self.dataclose[0])
            self.order = self.order_target_percent(target=0.00)

    def stop(self):
        # 新增加一个stop策略完成函数
        # 用于输出执行后数据
        self.log('(策略参数 bollwindow=%2d, devfactor=%.2f) ，最终资产总值： %.2f' %
                 (self.params.bollwindow, self.params.devfactor, self.broker.getvalue()), fgPrint=True)




