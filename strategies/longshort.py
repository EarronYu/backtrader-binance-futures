from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np
import backtrader as bt
import backtrader.indicators as btind
import datetime as dt
import pandas as pd
import pandas_datareader as web
from pandas import Series, DataFrame
import random
from copy import deepcopy
import optunity.metrics
import backtrader.analyzers as btanal


class long_short(bt.Strategy):
    """A simple moving average crossover strategy; crossing of a fast and slow moving average generates buy/sell
           signals"""
    params = dict(var1=20, var2=50)  # The windows for both var1 (fast) and var2 (slow) moving averages

    def __init__(self):
        """Initialize the strategy"""

        self.sma = dict()
        self.fastma = dict()
        self.slowma = dict()
        self.regime = dict()

        for d in self.getdatanames():
            # The moving averages
            self.sma[d] = btind.SMA(self.getdatabyname(d),  # The symbol for the moving average
                                    period=globalparams['sma_period'],  # Fast moving average
                                    plotname="SMA20: " + d)

    def next(self):
        """Define what will be done in a single step, including creating and closing trades"""
        for d in self.getdatanames():  # Looping through all symbols
            pos = self.getpositionbyname(d).size or 0
            if pos == 0:  # Are we out of the market?
                # Consider the possibility of entrance
                # Notice the indexing; [0] always means the present bar, and [-1] the bar immediately preceding
                # Thus, the condition below translates to: "If today the regime is bullish (greater than
                # 0) and yesterday the regime was not bullish"
                if self.sma[d][0] * self.params.var1 > self.getdatabyname(d).high[0]:  # A buy signal
                    self.order_target_percent(data=self.getdatabyname(d), target=0.98)

            else:  # We have an open position
                if self.getdatabyname(d).close[-1] * self.params.var2 <= self.getdatabyname(d).high[0]:  # A sell signal
                    self.order_target_percent(data=self.getdatabyname(d), target=0)
