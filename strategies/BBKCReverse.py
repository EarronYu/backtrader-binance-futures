import backtrader as bt
import functions.toolkit as tk
from datetime import datetime
import arrow


class BBKCReverse(bt.Strategy):
    """
    简单布林线策略
    价格突破上轨做多，下穿中轨平多
    价格突破下轨做空，上穿中轨平空
    """
    params = dict(window=20, bbdevs=2, kcdevs=1.5, bias_pct=0.005, volatile_pct=0.005, stop_loss=0.02, trail=False, )

    def log(self, txt, dt=None, fgPrint=False):
        # 增强型log记录函数，带fgPrint打印开关变量
        if fgPrint:
            dt = dt or self.datas[0].datetime.date(0)
            tn = tk.timNSec('', self.tim0wrk)
            # print('%s, %s，tn：%.2f' % (dt.isoformat(), txt))
            print('%s, %s，tim：%.2f s' % (dt.isoformat(), txt, tn))

    def __init__(self):
        # Set some pointers / references
        for i, d in enumerate(self.datas):
            if d._name == 'Real':
                self.real = d
            elif d._name == 'Heikin':
                self.ha = d

        self.buyprice = None
        self.buycomm = None
        self.sellprice = None
        self.sellcomm = None
        self.tim0wrk = arrow.now()
        self.dataclose = self.datas[0].close
        self.order = None

        bb = bt.ind.BollingerBands(self.ha, period=self.params.window, devfactor=self.params.bbdevs, movav=bt.ind.MovingAverageSimple, plot=False)
        ma = bt.ind.MovingAverageSimple(self.ha, period=self.p.window, plot=False)
        atr = self.params.kcdevs * bt.ind.ATR(self.ha, period=self.params.window, plot=False)
        kctop = ma + atr
        kcbot = ma - atr
        bbkctop = bt.If(bb.top > kctop, bb.top, kctop)
        bbkcbot = bt.If(bb.bot < kcbot, bb.bot, kcbot)

        volatile = abs(self.ha.open / ma - 1)
        high_volatile = bt.If(volatile >= self.params.volatile_pct, 1, 0)
        bias = abs(self.ha.close / ma - 1)
        high_bias = bt.If(bias >= self.params.bias_pct, 1, 0)

        crossupbot = bt.ind.CrossUp(self.ha.close, bbkcbot, plot=False)
        crossdowntop = bt.ind.CrossDown(self.ha.close, bbkctop, plot=False)
        self.close_long = bt.indicators.CrossUp(self.ha.close, bbkctop, plot=False)
        self.close_short = bt.indicators.CrossDown(self.ha.close, bbkcbot, plot=False)
        self.open_short = bt.And(crossdowntop, high_volatile, high_bias)
        self.open_long = bt.And(crossupbot, high_volatile, high_bias)

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
                # if not self.p.trail:
                #     stop_price = order.executed.price * (1.0 - self.p.stop_loss)
                #     self.sell(exectype=bt.Order.Stop, price=stop_price)
                # else:
                #     self.sell(exectype=bt.Order.StopTrail, trailamount=self.p.trail)
                self.log('买单执行BUY EXECUTED,成交价： %.2f,小计 Cost: %.2f,佣金 Comm %.2f'
                         % (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                # if not self.p.trail:
                #     stop_price = order.executed.price * (1.0 + self.p.stop_loss)
                #     self.buy(exectype=bt.Order.Stop, price=stop_price)
                # else:
                #     self.buy(exectype=bt.Order.StopTrail, trailamount=self.p.trail)
                self.log('卖单执行SELL EXECUTED,成交价： %.2f,小计 Cost: %.2f,佣金 Comm %.2f'
                         % (order.executed.price, order.executed.value, order.executed.comm))
                self.sellprice = order.executed.price
                self.sellcomm = order.executed.comm

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单Order： 取消Canceled/保证金Margin/拒绝Rejected')

        # 检查完成，没有交易中订单（pending order）
        # self.order = None

    def notify_trade(self, trade):
        # 检查交易trade是关闭
        if not trade.isclosed:
            return

        dt = self.data.datetime.date()
        self.log('---------------------------- TRADE ---------------------------------')
        self.log("1: Data Name:                            {}".format(trade.data._name))
        self.log("2: Bar Num:                              {}".format(len(trade.data)))
        self.log("3: Current date:                         {}".format(dt))
        self.log('4: Status:                               Trade Complete')
        self.log('5: Ref:                                  {}'.format(trade.ref))
        self.log('6: PnL:                                  {}'.format(round(trade.pnl, 2)))
        self.log('--------------------------------------------------------------------')

    def next(self):
        if self.order:
            return
        pfl = self.broker.get_cash()
        if self.open_long and not self.position:
            self.order = self.order_target_percent(data=self.real, target=0.99)
        if self.close_long and self.position:
            self.order = self.order_target_percent(data=self.real, target=0)
        if self.open_short and not self.position:
            self.order = self.order_target_percent(data=self.real, target=-0.99)
        if self.close_short and self.position:
            self.order = self.order_target_percent(data=self.real, target=0)

    def stop(self):
        # 新增加一个stop策略完成函数
        # 用于输出执行后数据
        self.log(
            f'(策略参数 window={self.params.window}, bbdevs={self.params.bbdevs}, kcdevs={self.params.kcdevs}, bias_pct={self.params.bias_pct}, volatile_pct='
            f'{self.params.volatile_pct}) ，最终资产总值：{self.broker.getvalue()}',
            fgPrint=True)
