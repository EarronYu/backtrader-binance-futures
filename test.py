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
        # todo 将这个策略参考SMAC和long——short进行修改

        self.params = dict()
        self.ind = dict()

        for d in self.getdatanames():
            if 'Heikin' in d:
                strategy_params = self.load_params(strategy=self.__class__.__name__.rstrip('_Heikin'), data=d)
                # self.params[d] = dict()
                self.params[d]['ema_fast_window'] = strategy_params['var1']
                self.params[d]['ema_slow_window'] = strategy_params['var2']
                self.ind[d]['stoploss'] = strategy_params['var3']

                self.ind[d] = dict()
                # 将指标的各项内容放到对应的数据源当中
                self.ind[d]['ema_fast'] = bt.indicators.EMA(self.getdatabyname(d),
                                                            period=self.params[d]['ema_fast_window'],
                                                            plotname="ema_fast: " + d)
                self.ind[d]['ema_slow'] = bt.indicators.EMA(self.getdatabyname(d),
                                                            period=self.params[d]['ema_slow_window'],
                                                            plotname="ema_slow: " + d)
                self.ind[d]['rsi'] = bt.indicators.RSI(self.getdatabyname(d))

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
            for d in self.getdatanames():
                if order.isbuy():
                    self.last_operation[d] = "long"
                    self.reset_order_indicators()
                    if ENV == DEVELOPMENT:
                        # print(order.executed.__dict__)
                        self.log('SHORT EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                                 (order.executed.price,
                                  order.executed.value,
                                  order.executed.comm), fgprint=False)
                    if ENV == PRODUCTION:
                        print(order.__dict__)
                        self.log('LONG EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                                 (order.ccxt_order['average'],
                                  float(order.ccxt_order['info']['cumQuote']),
                                  float(order.ccxt_order['info']['cumQuote']) * 0.0004), send_telegram=False,
                                 fgprint=False)
                        # tk.save_order(order, strategy=self.__class__.__name__, write=False)
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
                                  order.executed.comm), fgprint=False)
                    if ENV == PRODUCTION:
                        print(order.__dict__)
                        self.log('SHORT EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                                 (order.ccxt_order['average'],
                                  float(order.ccxt_order['info']['cumQuote']),
                                  float(order.ccxt_order['info']['cumQuote']) * 0.0004), send_telegram=False,
                                 fgprint=False)
                        # tk.save_order(order, strategy=self.__class__.__name__, write=False)
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
                self.profit[d] = 0
                # dt, dn = self.datetime.datetime(), d.__name
                # pos = self.getposition(d).size
                # 利用真实价格来计算PNL
                if self.buy_price_close[d] and self.buy_price_close[d] > 0:
                    self.profit[d] = float(self.getdatabyname(d.rstrip('_Heikin')).close[0] - self.buy_price_close[d]) / \
                                     self.buy_price_close[d]
                if self.sell_price_close[d] and self.sell_price_close[d] > 0:
                    self.profit[d] = float(
                        self.sell_price_close[d] - self.getdatabyname(d.rstrip('_Heikin')).close[0]) / \
                                     self.sell_price_close[d]

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
                    if self.ind[d]['rsi'][0] < 30 and self.ind[d]['ema_fast'][0] > self.ind[d]['ema_slow'][0]:
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
                        if -self.profit[d] > self.ind[d]['stoploss']:
                            self.log("STOP LOSS: percentage %.3f %%" % self.profit[d], fgprint=False)
                            if self.last_operation[d] == "long":
                                self.order = self.close_short(data=self.getdatabyname(d.rstrip('_Heikin')))
                            if self.last_operation[d] == "short":
                                self.order = self.close_short(data=self.getdatabyname(d.rstrip('_Heikin')))
                            # self.order = self.sell(d)
                            # print('卖卖卖')
                            # stop Loss