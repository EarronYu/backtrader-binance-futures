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
