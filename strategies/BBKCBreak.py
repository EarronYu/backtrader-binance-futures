import backtrader as bt
import functions.toolkit as tk
from datetime import datetime
import arrow


class BBKCBreak(bt.Strategy):
    """
    简单布林线策略
    价格突破上轨做多，下穿中轨平多
    价格突破下轨做空，上穿中轨平空
    """
    params = dict(window=262, bbdevs=3.7, kcdevs=1.37, bias_pct=0.398, volatile_pct=0.05)
    # window = 262, bbdevs = 3.702624414062499, kcdevs = 1.3728536286917288, bias_pct = 0.39861572204232243
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
                self.kl = d
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
        low_volatile = bt.If(volatile < 0.05, 1, 0)
        bias = abs(self.ha.close / ma - 1)
        low_bias = bt.If(bias <= self.params.bias_pct, 1, 0)

        crossuptop = bt.ind.CrossUp(self.ha.close, bbkctop, plot=False)
        crossdownbot = bt.ind.CrossDown(self.ha.close, bbkcbot, plot=False)
        self.close_long = bt.indicators.CrossDown(self.ha.close, bb.mid, plot=False)
        self.close_short = bt.indicators.CrossUp(self.ha.close, bb.mid, plot=False)
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

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单Order： 取消Canceled/保证金Margin/拒绝Rejected')

        # 检查完成，没有交易中订单（pending order）
        self.order = None

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
            self.order = self.order_target_percent(data=self.kl, target=0.99)
        if self.close_long and self.position:
            self.order = self.order_target_percent(data=self.kl, target=0)
        if self.open_short and not self.position:
            self.order = self.order_target_percent(data=self.kl, target=-0.99)
        if self.close_short and self.position:
            self.order = self.order_target_percent(data=self.kl, target=0)

    def stop(self):
        # 新增加一个stop策略完成函数
        # 用于输出执行后数据
        self.log(f'(策略参数 window={self.params.window}, bbdevs={self.params.bbdevs}, kcdevs={self.params.kcdevs}, bias_pct={self.params.bias_pct}) ，最终资产总值：{self.broker.getvalue()}', fgPrint=True)


class BBKCBreakWF(bt.Strategy):
    """The SMAC strategy but in a walk-forward analysis context"""
    params = {"start_dates": None,  # Starting days for trading periods (a list)
              "end_dates": None,  # Ending day for trading periods (a list)
              "var1": None,  # List of fast moving average windows, corresponding to start dates (a list)
              "var2": None}  # Like fast, but for slow moving average window (a list)

    # All the above lists must be of the same length, and they all line up

    def __init__(self):
        """Initialize the strategy"""
        self.sma = dict()
        self.var1 = dict()
        self.var2 = dict()
        self.regime = dict()

        self.date_combos = [c for c in zip(self.p.start_dates, self.p.end_dates)]
        # Error checking
        if type(self.p.start_dates) is not list or type(self.p.end_dates) is not list or \
                type(self.p.fast) is not list or type(self.p.slow) is not list:
            raise ValueError("Must past lists filled with numbers to params start_dates, end_dates, fast, slow.")
        elif len(self.p.start_dates) != len(self.p.end_dates) or \
                len(self.p.fast) != len(self.p.start_dates) or len(self.p.slow) != len(self.p.start_dates):
            raise ValueError("All lists passed to params must have same length.")
        for d in self.getdatanames():
            self.sma[d] = dict()
            self.var1[d] = dict()
            self.var2[d] = dict()
            self.regime[d] = dict()

            # Additional indexing, allowing for differing start/end dates
            for sd, ed, f, s in zip(self.p.start_dates, self.p.end_dates, self.p.var1, self.p.var2):
                # More error checking
                '''
                if type(f) is not int or type(s) is not int:
                    raise ValueError("Must include only integers in fast, slow.")
                elif f > s:
                    raise ValueError("Elements in fast cannot exceed elements in slow.")
                elif f <= 0 or s <= 0:
                    raise ValueError("Moving average windows must be positive.")


                if type(sd) is not dt.date or type(ed) is not dt.date:
                    raise ValueError("Only datetime dates allowed in start_dates, end_dates.")
                elif ed - sd < dt.timedelta(0):
                    raise ValueError("Start dates must always be before end dates.")
                '''
                # The moving averages
                # Notice that different moving averages are obtained for different combinations of
                # start/end dates
                self.sma[d][(sd, ed)] = btind.SimpleMovingAverage(self.getdatabyname(d),
                                                                  period=globalparams['sma_period'],
                                                                  plot=False)
                self.var1[d][(sd, ed)] = f
                self.var2[d][(sd, ed)] = s
                '''
                self.fastma[d][(sd, ed)] = btind.SimpleMovingAverage(self.getdatabyname(d),
                                                                     period=f,
                                                                     plot=False)
                self.slowma[d][(sd, ed)] = btind.SimpleMovingAverage(self.getdatabyname(d),
                                                                     period=s,
                                                                     plot=False)

                # Get the regime
                self.regime[d][(sd, ed)] = self.fastma[d][(sd, ed)] - self.slowma[d][(sd, ed)]
                # In the future, use the backtrader indicator btind.CrossOver()
                '''

    def next(self):
        """Define what will be done in a single step, including creating and closing trades"""

        # Determine which set of moving averages to use
        curdate = self.datetime.date(0)
        dtidx = None  # Will be index
        # Determine which period (if any) we are in
        for sd, ed in self.date_combos:
            # Debug output
            # print('{}: {} < {}: {}, {} < {}: {}'.format(
            #    len(self), sd, curdate, (sd <= curdate), curdate, ed, (curdate <= ed)))
            if sd <= curdate and curdate <= ed:
                dtidx = (sd, ed)
        # Debug output
        # print('{}: the dtixdx is {}, and curdate is {};'.format(len(self), dtidx, curdate))
        for d in self.getdatanames():  # Looping through all symbols
            pos = self.getpositionbyname(d).size or 0
            if dtidx is None:  # Not in any window
                break  # Don't engage in trades
            if pos == 0:  # Are we out of the market?
                # Consider the possibility of entrance
                # Notice the indexing; [0] always mens the present bar, and [-1] the bar immediately preceding
                # Thus, the condition below translates to: "If today the regime is bullish (greater than
                # 0) and yesterday the regime was not bullish"
                '''if self.slowma[d][dtidx][0] > self.getdatabyname(d).close[0]:  # A buy signal
                    self.sell(data=self.getdatabyname(d), size=1000)

            else:  # We have an open position
                if self.fastma[d][dtidx][0] < self.getdatabyname(d).close[0]:  # A sell signal
                    self.close(data=self.getdatabyname(d), size=1000)
            '''
                if self.sma[d][dtidx][0] * self.var1[d][dtidx] > self.getdatabyname(d).high[0]:  # A buy signal
                    self.order_target_percent(data=self.getdatabyname(d), target=0.98)

            else:  # We have an open position
                if self.getdatabyname(d).close[-1] * self.var2[d][dtidx] <= self.getdatabyname(d).high[0]:  # A sell signal
                    self.order_target_percent(data=self.getdatabyname(d), target=0)
