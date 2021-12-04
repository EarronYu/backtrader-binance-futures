#!/usr/bin/env python3

import backtrader as bt
import yaml
from config import DEVELOPMENT, COIN_TARGET, COIN_REFER, ENV, PRODUCTION, DEBUG

from strategies.base import StrategyBase

import toolkit as tk


class BasicRSI(StrategyBase):

    def __init__(self):
        # 含有Heikin的数据是存储所有策略信息的
        StrategyBase.__init__(self)
        self.log("Using RSI/EMA strategy", fgprint=False)
        # 每一次调用策略的时候会用策略去处理所有的策略参数，以及计算所有下属的时间区间，处理多个币种数据
        # self.lastRanks = []  # 上次交易股票的列表
        # 0号是指数，不进入选股池，从1号往后进入股票池
        # self.stocks = self.datas[1:]
        # todo 将
        
        self.params = dict()
        self.ind = dict()

        for d in self.getdatanames():
            if 'Heikin' in d:
                strategy_params = self.load_params(strategy=self.__class__.__name__.rstrip('_Heikin'), data=d)
                # self.params[d] = dict()
                self.params[d]['ema_fast_window'] = strategy_params['ema_fast_window']
                self.params[d]['ema_slow_window'] = strategy_params['ema_slow_window']
                # params = dict(
                #     rebal_monthday=[1],  # 每月1日执行再平衡
                #     num_volume=100,  # 成交量取前100名
                #     ema_fast_window=10,
                #     ema_slow_window=200
                # )
                # if i % 2 == 0:
                self.ind[d] = dict()
                # self.dict[d]['greater'] = self.datas[i+1]  # 取datas中基数的序列，具体取哪个要视datas当中的内容为准, 选出大周期
                # 将指标的各项内容放到对应的数据源当中
                self.ind[d]['ema_fast'] = bt.indicators.EMA(self.getdatabyname(d), period=self.params[d]['ema_fast_window'], plotname="ema_fast: " + d)
                self.ind[d]['ema_slow'] = bt.indicators.EMA(self.getdatabyname(d), period=self.params[d]['ema_slow_window'], plotname="ema_slow: " + d)
                self.ind[d]['rsi'] = bt.indicators.RSI(self.getdatabyname(d))

        # 定时器
        # self.add_timer(
        #     when=bt.Timer.SESSION_START,
        #     monthdays=self.p.rebal_monthday,  # 每月1号触发再平衡
        #     monthcarry=True,  # 若再平衡日不是交易日，则顺延触发notify_timer
        # )

        # self.ema_fast = bt.indicators.EMA(period=self.p.period_ema_fast)
        # self.ema_slow = bt.indicators.EMA(period=self.p.period_ema_slow)
        # self.rsi = bt.indicators.RelativeStrengthIndex()
        # self.ema_fast.plotinfo.plot = False

    # def notify_timer(self, timer, when, *args, **kwargs):
    #     # 只在5，9，11月的1号执行再平衡
    #     if self.data0.datetime.date(0).month in [5, 9, 11]:
    #         self.rebalance_portfolio()  # 执行再平衡

    def load_params(self, strategy, data):
        with open('../dataset/symbol_config.yaml', 'r') as f:
            symbol_config = f.read()
            symbol_config = yaml.load(symbol_config, Loader=yaml.FullLoader)
            param = data.replace('USDT', f'USDT_{strategy}')
            # BNBUSDT_MyStrategy_10m
            # param = param.remove('_Kline')
            param = param.split('_')
            strategy_params = symbol_config[param[0]][param[1]][param[2]]
            f.close()
        return strategy_params

    def notify_order(self, order):
        StrategyBase.notify_order(self, order)
        if order.status in [order.Completed]:
            # 检查订单order是否完成
            # 注意: 如果现金不足，经纪人broker会拒绝订单reject order
            # 可以修改相关参数，调整进行空头交易
            for i, d in enumerate(self.datas):
                if order.isbuy():
                    self.last_operation[d] = "long"
                    self.reset_order_indicators()
                    if ENV == DEVELOPMENT:
                        # print(order.executed.__dict__)
                        self.log('SHORT EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                                 (order.executed.price,
                                  order.executed.value,
                                  order.executed.comm), fgprint=True)
                    if ENV == PRODUCTION:
                        print(order.__dict__)
                        self.log('LONG EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                                 (order.ccxt_order['average'],
                                  float(order.ccxt_order['info']['cumQuote']),
                                  float(order.ccxt_order['info']['cumQuote']) * 0.0004), send_telegram=False, fgprint=True)
                        tk.save_order(order, strategy=self.__class__.__name__, write=False)
                        # self.buyprice = order.ccxt_order['average']
                        # self.buycomm = order.ccxt_order['cost'] * 0.0004
                    if ENV == DEVELOPMENT:
                        print('order info: ', order.__dict__)
                if order.issell():
                    self.last_operation[d] = "short"
                    self.reset_order_indicators()
                    if ENV == DEVELOPMENT:
                        # print(order.executed.__dict__)
                        self.log('SHORT EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                                 (order.executed.price,
                                  order.executed.value,
                                  order.executed.comm), fgprint=True)
                    if ENV == PRODUCTION:
                        print(order.__dict__)
                        self.log('SHORT EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                                 (order.ccxt_order['average'],
                                  float(order.ccxt_order['info']['cumQuote']),
                                  float(order.ccxt_order['info']['cumQuote']) * 0.0004), send_telegram=False, fgprint=True)
                        tk.save_order(order, strategy=self.__class__.__name__, write=False)
                        # self.sellprice = order.ccxt_order['average']
                        # self.sellcomm = order.ccxt_order['cost'] * 0.0004
                    if ENV == DEVELOPMENT:
                        print('order info: ', order.__dict__)
                self.bar_executed = len(self)
        # Sentinel to None: new orders allowed
        self.order = None

    def update_indicators(self):
        # self.profit = dict()
        for d in self.getdatanames():
            if 'Heikin' in d:
                # d = d.remove('_Heikin')  # 以策略_周期这样干干净净的形式来作为其参数的容器
                self.profit[d] = 0
                # dt, dn = self.datetime.datetime(), d.__name
                # pos = self.getposition(d).size
                # 利用真实价格来计算PNL
                if self.buy_price_close[d] and self.buy_price_close[d] > 0:
                    self.profit[d] = float(self.getdatabyname(d.rstrip('_Heikin')).close[0] - self.buy_price_close[d]) / self.buy_price_close[d]
                if self.sell_price_close[d] and self.sell_price_close[d] > 0:
                    self.profit[d] = float(self.sell_price_close[d] - self.getdatabyname(d.rstrip('_Heikin')).close[0]) / self.sell_price_close[d]

    def next(self):
        self.update_indicators()
        if self.status != "LIVE" and ENV == PRODUCTION:  # waiting for live status in production
            return
        if self.order:  # waiting for pending order
            return
        for d in self.getdatanames():
            if 'Heikin' in d:
                # d = d.rstrp('_Heikin')  # 以策略_周期这样干干净净的形式来作为其参数的容器,代表真实数据
                # 永远以真实数据来进行下单操作
                # dt, dn = self.datetime.datetime(), d.__name
                pos = self.getpositionbyname(d).size or 0
                # if not pos:
                # self.order = self.buy(d)
                # print(f'买买买{dt}, {dn}')
                if self.last_operation[d] != "long":
                    if self.ind[d]['rsi'] < 30 and self.ind[d]['ema_fast'] > self.ind[d]['ema_slow']:
                        if not pos:
                            self.order = self.long(data=self.getdatabyname(d.rstrip('_Heikin')))
                        else:
                            self.order = self.close(data=self.getdatabyname(d.rstrip('_Heikin')))
                            self.order = self.long(data=self.getdatabyname(d.rstrip('_Heikin')))
                if self.last_operation[d] != "short":
                    if self.rsi > 70:
                        if not pos:
                            self.order = self.short(data=self.getdatabyname(d.rstrip('_Heikin')))
                        else:
                            self.order = self.close(data=self.getdatabyname(d.rstrip('_Heikin')))
                            self.order = self.short(data=self.getdatabyname(d.rstrip('_Heikin')))
                else:
                    if pos:
                        if self.profit[d] < -0.03:
                            self.log("STOP LOSS: percentage %.3f %%" % self.profit[d], fgprint=False)
                            if self.last_operation[d] == "long":
                                self.order = self.close_short(data=self.getdatabyname(d.rstrip('_Heikin')))
                            if self.last_operation[d] == "short":
                                self.order = self.close_short(data=self.getdatabyname(d.rstrip('_Heikin')))
                            # self.order = self.sell(d)
                            # print('卖卖卖')
                            # stop Loss
    # 以下的交易池的内容后面再来写
    # def rebalance_portfolio(self):
    #     # 从指数取得当前日期
    #     self.currDate = self.data0.datetime.date(0)
    #     print('rebalance_portfolio currDate', self.currDate, len(self.stocks))
    #
    #     # 如果是指数的最后一本bar，则退出，防止取下一日开盘价越界错
    #     if len(self.datas[0]) == self.data0.buflen():
    #         return
    #
    #     # 取消以往所下订单（已成交的不会起作用）
    #     # for o in self.order_list:
    #     #     self.cancel(o)
    #     # self.order_list = []  # 重置订单列表
    #
    #     # for d in self.stocks:
    #     #     print('sma', d._name, self.sma[d][0],self.sma[d][1], d.marketdays[0])
    #
    #     # 最终标的选取过程
    #     # 1 先做排除筛选过程
    #     self.ranks = [d for d in self.stocks if
    #                   len(d) > 0  # 重要，到今日至少要有一根实际bar
    #                   and d.marketdays > 3 * 365  # 到今天至少上市
    #                   # 今日未停牌 (若去掉此句，则今日停牌的也可能进入，并下订单，次日若复牌，则次日可能成交）（假设原始数据中已删除无交易的记录)
    #                   and d.datetime.date(0) == self.currDate
    #                   and d.roe >= 0.1
    #                   and 100 > d.pe > 0
    #                   and len(d) >= self.p.period  # 最小期，至少需要5根bar
    #                   and d.close[0] > self.sma[d][1]
    #                   ]
    #
    #     # 2 再做排序挑选过程
    #     self.ranks.sort(key=lambda d: d.volume, reverse=True)  # 按成交量从大到小排序
    #     self.ranks = self.ranks[0:self.p.num_volume]  # 取前num_volume名
    #
    #     if len(self.ranks) == 0:  # 无股票选中，则返回
    #         return
    #
    #     # 3 以往买入的标的，本次不在标的中，则先平仓
    #     data_toclose = set(self.lastRanks) - set(self.ranks)
    #     for d in data_toclose:
    #         print('sell 平仓', d._name, self.getposition(d).size)
    #         o = self.close(data=d)
    #         self.order_list.append(o)  # 记录订单
    #
    #     # 4 本次标的下单
    #     # 每只股票买入资金百分比，预留2%的资金以应付佣金和计算误差
    #     buypercentage = (1 - 0.02) / len(self.ranks)
    #
    #     # 得到目标市值
    #     targetvalue = buypercentage * self.broker.getvalue()
    #     # 为保证先卖后买，股票要按持仓市值从大到小排序
    #     self.ranks.sort(key=lambda d: self.broker.getvalue([d]), reverse=True)
    #     self.log('下单, 标的个数 %i, targetvalue %.2f, 当前总市值 %.2f' %
    #              (len(self.ranks), targetvalue, self.broker.getvalue()))
    #
    #     for d in self.ranks:
    #         # 按次日开盘价计算下单量，下单量是100的整数倍
    #         size = int(
    #             abs((self.broker.getvalue([d]) - targetvalue) / d.open[1] // 100 * 100))
    #         validday = d.datetime.datetime(1)  # 该股下一实际交易日
    #         if self.broker.getvalue([d]) > targetvalue:  # 持仓过多，要卖
    #             # 次日跌停价近似值
    #             lowerprice = d.close[0] * 0.9 + 0.02
    #
    #             o = self.sell(data=d, size=size, exectype=bt.Order.Limit,
    #                           price=lowerprice, valid=validday)
    #         else:  # 持仓过少，要买
    #             # 次日涨停价近似值
    #             upperprice = d.close[0] * 1.1 - 0.02
    #             o = self.buy(data=d, size=size, exectype=bt.Order.Limit,
    #                          price=upperprice, valid=validday)
    #
    #         self.order_list.append(o)  # 记录订单
    #
    #     self.lastRanks = self.ranks  # 跟踪上次买入的标的


class BasicRSIWalkForward(bt.Strategy):  # todo 1202 I'm working on this to turn it into a strategy competitive to WF method
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
