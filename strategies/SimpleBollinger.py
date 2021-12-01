import backtrader as bt
import functions.toolkit as tk
from datetime import datetime
import arrow


class SimpleBollinger(bt.Strategy):
    """
    简单布林线策略
    价格突破上轨做多，下穿中轨平多
    价格突破下轨做空，上穿中轨平空
    """
    params = dict(bollwindow=200, devfactor=2)

    def log(self, txt, dt=None, fgPrint=False):
        # 增强型log记录函数，带fgPrint打印开关变量
        if fgPrint:
            dt = dt or self.datas[0].datetime.date(0)
            tn = tk.timNSec('', self.tim0wrk)
            # print('%s, %s，tn：%.2f' % (dt.isoformat(), txt))
            print('%s, %s，tim：%.2f s' % (dt.isoformat(), txt, tn))

    def __init__(self):
        self.buyprice = None
        self.buycomm = None
        self.sellprice = None
        self.sellcomm = None
        self.tim0wrk = arrow.now()
        self.dataclose = self.datas[0].close
        self.order = None

        midband = bt.indicators.MovingAverageSimple(self.data, period=self.params.bollwindow)
        std = bt.indicators.StandardDeviation(self.data, period=self.params.bollwindow, plot=False)
        topband = midband + self.params.devfactor * std
        botband = midband - self.params.devfactor * std
        self.open_short = bt.indicators.CrossUp(self.data.close, topband, plot=False)
        self.close_long = bt.indicators.CrossDown(self.data.close, midband, plot=False)
        self.open_long = bt.indicators.CrossDown(self.data.close, botband, plot=False)
        self.close_short = bt.indicators.CrossUp(self.data.close, midband, plot=False)

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

            self.bar_executed = len(self)

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
        pfl = self.broker.get_cash()
        if self.open_long and not self.position:
            self.log('设置多单 BUY CREATE, %.2f, name : %s' % (self.dataclose[0], self.datas[0]._name))
            self.order = self.order_target_percent(target=0.99)
        if self.close_long and self.position:
            self.log('设置平多单 CLOSE LONG CREATE, %.2f, name : %s' % (self.dataclose[0], self.datas[0]._name))
            self.order = self.order_target_percent(target=0)
        if self.open_short and not self.position:
            self.log('设置空单 BUY CREATE, %.2f, name : %s' % (self.dataclose[0], self.datas[0]._name))
            self.order = self.order_target_percent(target=-0.99)
        if self.close_short and self.position:
            self.log('设置平空单 CLOSE LONG CREATE, %.2f, name : %s' % (self.dataclose[0], self.datas[0]._name))
            self.order = self.order_target_percent(target=0)

    def stop(self):
        # 新增加一个stop策略完成函数
        # 用于输出执行后数据
        self.log('(策略参数 bollwindow=%2d, devfactor=%.2f) ，最终资产总值： %.2f' %
                 (self.params.bollwindow, self.params.devfactor, self.broker.getvalue()), fgPrint=True)
