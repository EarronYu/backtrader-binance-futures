#!/usr/bin/env python3

import backtrader as bt

from config import DEVELOPMENT, COIN_TARGET, COIN_REFER, ENV, PRODUCTION, DEBUG

from strategies.base import StrategyBase


class BasicRSI(StrategyBase):
    params = dict(
        period_ema_fast=10,
        period_ema_slow=200
    )

    def __init__(self):
        StrategyBase.__init__(self)
        self.log("Using RSI/EMA strategy", fgprint=False)

        self.ema_fast = bt.indicators.EMA(period=self.p.period_ema_fast)
        self.ema_slow = bt.indicators.EMA(period=self.p.period_ema_slow)
        self.rsi = bt.indicators.RelativeStrengthIndex()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 检查订单执行状态order.status：
            # Buy/Sell order submitted/accepted to/by broker
            # broker经纪人：submitted提交/accepted接受,Buy买单/Sell卖单
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            self.log('ORDER ACCEPTED/SUBMITTED', fgprint=True)
            self.order = order
            return

        if order.status in [order.Expired]:
            self.log('LONG EXPIRED', send_telegram=False, fgprint=True)

        if order.status in [order.Completed]:
            # 检查订单order是否完成
            # 注意: 如果现金不足，经纪人broker会拒绝订单reject order
            # 可以修改相关参数，调整进行空头交易
            if order.isbuy():
                self.last_operation = "long"
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
                              order.ccxt_order['cost'],
                              order.ccxt_order['cost'] * 0.0004), send_telegram=False, fgprint=True)
                    # self.buyprice = order.ccxt_order['average']
                    # self.buycomm = order.ccxt_order['cost'] * 0.0004
                if ENV == DEVELOPMENT:
                    print('order info: ', order.__dict__)
            if order.issell():
                self.last_operation = "short"
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
                              order.ccxt_order['cost'],
                              order.ccxt_order['cost'] * 0.0004), send_telegram=False, fgprint=True)
                    # self.sellprice = order.ccxt_order['average']
                    # self.sellcomm = order.ccxt_order['cost'] * 0.0004
                if ENV == DEVELOPMENT:
                    print('order info: ', order.__dict__)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected: Status %s - %s' % (order.Status[order.status],
                                                                         self.last_operation), send_telegram=False,
                     fgprint=True)
        # Sentinel to None: new orders allowed
        self.order = None

    def next(self):
        self.update_indicators()

        if self.status != "LIVE" and ENV == PRODUCTION:  # waiting for live status in production
            return

        if self.order:  # waiting for pending order
            return

        # stop Loss
        if self.profit < -0.03:
            self.log("STOP LOSS: percentage %.3f %%" % self.profit, fgprint=False)
            if self.last_operation == "long":
                self.order = self.close_short()
            if self.last_operation == "short":
                self.order = self.close_short()

        if self.last_operation != "long":
            if self.rsi < 30 and self.ema_fast > self.ema_slow:
                if not self.position:
                    self.order = self.long()
                else:
                    self.order = self.close()
                    self.order = self.long()

        if self.last_operation != "short":
            if self.rsi > 70:
                if not self.position:
                    self.order = self.short()
                else:
                    self.order = self.close()
                    self.order = self.short()
