#!/usr/bin/env python3

from datetime import datetime
import backtrader as bt
from termcolor import colored
from config import DEVELOPMENT, COIN_TARGET, COIN_REFER, ENV, PRODUCTION, DEBUG
from utils import send_telegram_message


class StrategyBase(bt.Strategy):
    def __init__(self):
        # Set some pointers / references
        for i, d in enumerate(self.datas):
            if d._name == 'Kline':
                self.kl = d
            elif d._name == 'Heikin':
                self.ha = d

        self.buyprice = None
        self.buycomm = None
        self.sellprice = None
        self.sellcomm = None

        self.order = None
        self.last_operation = None
        self.status = "DISCONNECTED"
        self.bar_executed = 0

        self.buy_price_close = None
        self.sell_price_close = None

        self.soft_sell = False
        self.hard_sell = False
        self.soft_buy = False
        self.hard_buy = False

        self.profit = 0

        self.log("Base strategy initialized", fgprint=True)

    def reset_order_indicators(self):
        self.soft_sell = False
        self.hard_sell = False
        self.buy_price_close = None
        self.sell_price_close = None
        self.soft_buy = False
        self.hard_buy = False

    def notify_data(self, data, status, *args, **kwargs):
        self.status = data._getstatusname(status)
        print(self.status)
        if status == data.LIVE:
            self.log("LIVE DATA - Ready to trade", fgprint=True)

    def update_indicators(self):
        self.profit = 0
        if self.buy_price_close and self.buy_price_close > 0:
            self.profit = float(self.kl.close[0] - self.buy_price_close) / self.buy_price_close
        if self.sell_price_close and self.sell_price_close > 0:
            self.profit = float(self.sell_price_close - self.kl.close[0]) / self.sell_price_close

    def short(self):
        if self.last_operation == "short":
            return

        self.sell_price_close = self.kl.close[0]
        price = self.kl.close[0]

        if ENV == DEVELOPMENT:
            self.log("open short ordered: $%.2f" % self.kl.close[0], fgprint=True)
            return self.sell()

        cash, value = self.broker.get_wallet_balance(COIN_REFER)
        # print(cash, ' ', value)
        amount = (value / price) * 0.99
        self.log("open short ordered: $%.2f. Amount %.6f %s. Balance $%.2f USDT" % (self.kl.close[0],
                                                                              amount, COIN_TARGET, value),
                 send_telegram=False, fgprint=True)
        return self.sell(data=self.kl, size=amount)

    def long(self):
        if self.last_operation == "long":
            return

        # self.log("Buy ordered: $%.2f" % self.data0.close[0], True)
        self.buy_price_close = self.kl.close[0]
        price = self.kl.close[0]

        if ENV == DEVELOPMENT:
            self.log("open long ordered: $%.2f" % self.kl.close[0], fgprint=True)
            return self.buy()

        cash, value = self.broker.get_wallet_balance(COIN_REFER)
        amount = (value / price) * 0.99  # Workaround to avoid precision issues
        self.log("open long ordered: $%.2f. Amount %.6f %s. Balance $%.2f USDT" % (self.kl.close[0],
                                                                             amount, COIN_TARGET, value),
                 send_telegram=False, fgprint=True)
        return self.buy(data=self.kl, size=amount)

    def close_long(self):
        if self.last_operation == "long":
            if ENV == DEVELOPMENT:
                self.log("close long ordered: $%.2f" % self.kl.close[0], fgprint=True)
                return self.close()
            self.log("close long ordered: $%.2f" % self.kl.close[0], send_telegram=False, fgprint=True)
            return self.close()

    def close_short(self):
        if self.last_operation == "short":
            return self.close()

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
                    # print(order.__dict__)
                    self.log('LONG EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                             (order.ccxt_order['average'],
                              order.ccxt_order['cost'],
                              order.ccxt_order['cost'] * 0.0004), send_telegram=False, fgprint=True)
                    self.buyprice = order.ccxt_order['average']
                    self.buycomm = order.ccxt_order['cost'] * 0.0004
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
                    # print(order.__dict__)
                    self.log('SHORT EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                             (order.ccxt_order['average'],
                              order.ccxt_order['cost'],
                              order.ccxt_order['cost'] * 0.0004), send_telegram=False, fgprint=True)
                    self.sellprice = order.ccxt_order['average']
                    self.sellcomm = order.ccxt_order['cost'] * 0.0004
                if ENV == DEVELOPMENT:
                    print('order info: ', order.__dict__)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected: Status %s - %s' % (order.Status[order.status],
                                                                         self.last_operation), send_telegram=False,
                     fgprint=True)
        # Sentinel to None: new orders allowed
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        color = 'green'
        if trade.pnl < 0:
            color = 'red'

        self.log(colored('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm), color),
                 send_telegram=False,
                 fgprint=True)

    def log(self, txt, send_telegram=False, color=None, fgprint=True):
        if fgprint:
            value = datetime.now()
            if len(self) > 0:
                value = self.kl.datetime.datetime()
            if color:
                txt = colored(txt, color)
            print('[%s] %s' % (value.strftime("%d-%m-%y %H:%M"), txt))
        if send_telegram:
            send_telegram_message(txt)
