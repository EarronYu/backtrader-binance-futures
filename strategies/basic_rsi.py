#!/usr/bin/env python3

import backtrader as bt
import yaml
from config import DEVELOPMENT, COIN_TARGET, COIN_REFER, ENV, PRODUCTION, DEBUG

from strategies.base import StrategyBase

import toolkit as tk


class BasicRSI(StrategyBase):

    def __init__(self):
        StrategyBase.__init__(self)
        self.log("Using RSI/EMA strategy", fgprint=False)
        # 每一次调用策略的时候会用策略去处理所有的策略参数，以及计算所有下属的时间区间，处理多个币种数据
        self.lastRanks = []  # 上次交易股票的列表
        # 0号是指数，不进入选股池，从1号往后进入股票池
        self.stocks = self.datas[1:]

        self.params = dict()
        self.ind = dict()
        for i, d in enumerate(self.datas):
            # if i % 2 == 0:
            strategy_params = self.uodate_params(strategy=self.__class__.__name__, data=d)
            self.params[d] = dict()
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
            self.ind[d]['ema_fast'] = bt.indicators.EMA(data=d, period=self.params[d]['ema_fast_window'])
            self.ind[d]['ema_slow'] = bt.indicators.EMA(period=self.params[d]['ema_slow_window'])
            self.ind[d]['rsi'] = bt.indicators.RelativeStrengthIndex()

        # 定时器
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=self.p.rebal_monthday,  # 每月1号触发再平衡
            monthcarry=True,  # 若再平衡日不是交易日，则顺延触发notify_timer
        )

        # self.ema_fast = bt.indicators.EMA(period=self.p.period_ema_fast)
        # self.ema_slow = bt.indicators.EMA(period=self.p.period_ema_slow)
        # self.rsi = bt.indicators.RelativeStrengthIndex()
        # self.ema_fast.plotinfo.plot = False

    # def notify_timer(self, timer, when, *args, **kwargs):
    #     # 只在5，9，11月的1号执行再平衡
    #     if self.data0.datetime.date(0).month in [5, 9, 11]:
    #         self.rebalance_portfolio()  # 执行再平衡

    def uodate_params(self, strategy, data):
        with open('../dataset/symbol_config.yaml', 'r') as f:
            symbol_config = f.read()
            symbol_config = yaml.load(symbol_config, Loader=yaml.FullLoader)
            param = data.replace('USDT', f'USDT_{strategy}')
            param = param.remove('_Kline')
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
        for i, d in enumerate(self.datas):
            self.profit[d] = 0
            """
            这里还需要改
            """
            # dt, dn = self.datetime.datetime(), d.__name
            # pos = self.getposition(d).size
            if self.buy_price_close[d] and self.buy_price_close[d] > 0:
                self.profit[d] = float(self.d.close[0] - self.buy_price_close[d]) / self.buy_price_close[d]
            if self.sell_price_close[d] and self.sell_price_close[d] > 0:
                self.profit[d] = float(self.sell_price_close[d] - self.d.close[0]) / self.sell_price_close[d]

    def next(self):
        self.update_indicators()
        if self.status != "LIVE" and ENV == PRODUCTION:  # waiting for live status in production
            return
        if self.order:  # waiting for pending order
            return
        for i, d in enumerate(self.datas):
            # dt, dn = self.datetime.datetime(), d.__name
            pos = self.getposition(d).size
            # if not pos:
            # self.order = self.buy(d)
            # print(f'买买买{dt}, {dn}')
            if self.last_operation[d] != "long":
                if self.ind[d]['rsi'] < 30 and self.ind[d]['ema_fast'] > self.ind[d]['ema_slow']:
                    if not pos:
                        self.order = self.long(d)
                    else:
                        self.order = self.close(d)
                        self.order = self.long(d)
            if self.last_operation[d] != "short":
                if self.rsi > 70:
                    if not pos:
                        self.order = self.short(d)
                    else:
                        self.order = self.close(d)
                        self.order = self.short(d)
            else:
                if pos:
                    if self.profit[d] < -0.03:
                        self.log("STOP LOSS: percentage %.3f %%" % self.profit[d], fgprint=False)
                        if self.last_operation[d] == "long":
                            self.order = self.close_short(d)
                        if self.last_operation[d] == "short":
                            self.order = self.close_short(d)
                        # self.order = self.sell(d)
                        # print('卖卖卖')
                        # stop Loss

    def rebalance_portfolio(self):
        # 从指数取得当前日期
        self.currDate = self.data0.datetime.date(0)
        print('rebalance_portfolio currDate', self.currDate, len(self.stocks))

        # 如果是指数的最后一本bar，则退出，防止取下一日开盘价越界错
        if len(self.datas[0]) == self.data0.buflen():
            return

        # 取消以往所下订单（已成交的不会起作用）
        # for o in self.order_list:
        #     self.cancel(o)
        # self.order_list = []  # 重置订单列表

        # for d in self.stocks:
        #     print('sma', d._name, self.sma[d][0],self.sma[d][1], d.marketdays[0])

        # 最终标的选取过程
        # 1 先做排除筛选过程
        self.ranks = [d for d in self.stocks if
                      len(d) > 0  # 重要，到今日至少要有一根实际bar
                      and d.marketdays > 3 * 365  # 到今天至少上市
                      # 今日未停牌 (若去掉此句，则今日停牌的也可能进入，并下订单，次日若复牌，则次日可能成交）（假设原始数据中已删除无交易的记录)
                      and d.datetime.date(0) == self.currDate
                      and d.roe >= 0.1
                      and 100 > d.pe > 0
                      and len(d) >= self.p.period  # 最小期，至少需要5根bar
                      and d.close[0] > self.sma[d][1]
                      ]

        # 2 再做排序挑选过程
        self.ranks.sort(key=lambda d: d.volume, reverse=True)  # 按成交量从大到小排序
        self.ranks = self.ranks[0:self.p.num_volume]  # 取前num_volume名

        if len(self.ranks) == 0:  # 无股票选中，则返回
            return

        # 3 以往买入的标的，本次不在标的中，则先平仓
        data_toclose = set(self.lastRanks) - set(self.ranks)
        for d in data_toclose:
            print('sell 平仓', d._name, self.getposition(d).size)
            o = self.close(data=d)
            self.order_list.append(o)  # 记录订单

        # 4 本次标的下单
        # 每只股票买入资金百分比，预留2%的资金以应付佣金和计算误差
        buypercentage = (1 - 0.02) / len(self.ranks)

        # 得到目标市值
        targetvalue = buypercentage * self.broker.getvalue()
        # 为保证先卖后买，股票要按持仓市值从大到小排序
        self.ranks.sort(key=lambda d: self.broker.getvalue([d]), reverse=True)
        self.log('下单, 标的个数 %i, targetvalue %.2f, 当前总市值 %.2f' %
                 (len(self.ranks), targetvalue, self.broker.getvalue()))

        for d in self.ranks:
            # 按次日开盘价计算下单量，下单量是100的整数倍
            size = int(
                abs((self.broker.getvalue([d]) - targetvalue) / d.open[1] // 100 * 100))
            validday = d.datetime.datetime(1)  # 该股下一实际交易日
            if self.broker.getvalue([d]) > targetvalue:  # 持仓过多，要卖
                # 次日跌停价近似值
                lowerprice = d.close[0] * 0.9 + 0.02

                o = self.sell(data=d, size=size, exectype=bt.Order.Limit,
                              price=lowerprice, valid=validday)
            else:  # 持仓过少，要买
                # 次日涨停价近似值
                upperprice = d.close[0] * 1.1 - 0.02
                o = self.buy(data=d, size=size, exectype=bt.Order.Limit,
                             price=upperprice, valid=validday)

            self.order_list.append(o)  # 记录订单

        self.lastRanks = self.ranks  # 跟踪上次买入的标的
