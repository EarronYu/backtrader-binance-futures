#!/usr/bin/env python3

from datetime import datetime
import backtrader as bt
from termcolor import colored
from config import DEVELOPMENT, COIN_TARGET, COIN_REFER, ENV, PRODUCTION, DEBUG
from utils import send_telegram_message


class StrategyBase(bt.Strategy):
    def __init__(self):
        # # Set some pointers / references
        # for i, d in enumerate(self.datas):
        #     if d._name == 'Kline':
        #         self.kl = d
        #     elif d._name == 'Heikin':
        #         self.ha = d

        self.buyprice = None
        self.buycomm = None
        self.sellprice = None
        self.sellcomm = None

        self.order = None
        self.last_operation = dict()
        self.status = "DISCONNECTED"
        self.bar_executed = 0

        self.buy_price_close = dict()
        self.sell_price_close = dict()

        self.soft_sell = False
        self.hard_sell = False
        self.soft_buy = False
        self.hard_buy = False

        self.profit = dict()

        self.log("Base strategy initialized", fgprint=True)

    def reset_order_indicators(self):
        self.soft_sell = False
        self.hard_sell = False
        self.buy_price_close = dict()
        self.sell_price_close = dict()
        self.soft_buy = False
        self.hard_buy = False

    def notify_data(self, data, status, *args, **kwargs):
        self.status = data._getstatusname(status)
        print(self.status)
        if status == data.LIVE:
            self.log("LIVE DATA - Ready to trade", fgprint=True)

    def notify_order(self, order):
        # StrategyBase.notify_order(self, order)
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

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # return
            for i, d in enumerate(self.datas):
                self.log('Order Canceled/Margin/Rejected: Status %s - %s' % (order.Status[order.status],
                                                                             self.last_operation[d]), send_telegram=False,
                         fgprint=True)

    def short(self, data=None):
        if self.last_operation[data] == "short":
            return
        if isinstance(data, str):
            data = self.getdatabyname(data)
        data = data if data is not None else self.datas[0]
        self.sell_price_close[data] = data.close[0]
        price = data.close[0]

        if ENV == DEVELOPMENT:
            self.log("open short ordered: $%.2f" % data.close[0], fgprint=True)
            return self.sell(data=data)

        cash, value = self.broker.get_wallet_balance(COIN_REFER)
        # print(cash, ' ', value)
        amount = (value / price) * 0.99
        self.log("open short ordered: $%.2f. Amount %.6f %s. Balance $%.2f USDT" % (data.close[0],
                                                                              amount, COIN_TARGET, value),
                 send_telegram=False, fgprint=True)
        return self.sell(data=data, size=amount)

    def long(self, data=None):
        if self.last_operation[data] == "long":
            return
        if isinstance(data, str):
            data = self.getdatabyname(data)
        data = data if data is not None else self.datas[0]
        # self.log("Buy ordered: $%.2f" % self.data0.close[0], True)
        self.buy_price_close[data] = data.close[0]
        price = data.close[0]

        if ENV == DEVELOPMENT:
            self.log("open long ordered: $%.2f" % data.close[0], fgprint=True)
            return self.buy(data=data)

        cash, value = self.broker.get_wallet_balance(COIN_REFER)
        amount = (value / price) * 0.99  # Workaround to avoid precision issues
        self.log("open long ordered: $%.2f. Amount %.6f %s. Balance $%.2f USDT" % (data.close[0],
                                                                             amount, COIN_TARGET, value),
                 send_telegram=False, fgprint=True)
        return self.buy(data=data, size=amount)

    def close_long(self, data=None):
        if isinstance(data, str):
            data = self.getdatabyname(data)
        elif data is None:
            data = self.data
        if self.last_operation[data] == "long":
            if ENV == DEVELOPMENT:
                self.log("close long ordered: $%.2f" % data.close[0], fgprint=True)
                return self.close(data=data)
            self.log("close long ordered: $%.2f" % data.close[0], send_telegram=False, fgprint=True)
            return self.close(data=data)

    def close_short(self, data=None):
        if isinstance(data, str):
            data = self.getdatabyname(data)
        elif data is None:
            data = self.data
        if self.last_operation[data] == "short":
            if ENV == DEVELOPMENT:
                self.log("close short ordered: $%.2f" % data.close[0], fgprint=True)
                return self.close(data=data)
            self.log("close short ordered: $%.2f" % data.close[0], send_telegram=False, fgprint=True)
            return self.close(data=data)

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
