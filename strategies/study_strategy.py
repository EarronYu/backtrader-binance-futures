import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import math


class MyStrategy(bt.Strategy):
    params = dict(period1=20, period2=25, period3=10, period4=10)
    # tuple format
    # params = (('period', 20),)

    def __init__(self):
        """
        The self.datas array items can be directly accessed with additional automatic member variables:
            self.data targets self.datas[0]
            self.dataX targets self.datas[X]
        """
        sma1 = btind.SimpleMovingAverage(self.datas[0], period=self.params.period)
        # simplified
        # sma = btind.SimpleMovingAverage(period=self.params.period)
        # This 2nd Moving Average operates using sma1 as "data"
        sma2 = btind.SimpleMovingAverage(sma1, period=self.p.period2)
        # New data created via arithmetic operation
        something = sma2 - sma1 + self.data.close[0]
        # This 3rd Moving Average operates using something  as "data"
        sma3 = btind.SimpleMovingAverage(something, period=self.p.period3)
        # Comparison operators work too ...
        greater = sma3 > sma1
        # Pointless Moving Average of True/False values but valid
        # This 4th Moving Average operates using greater  as "data"
        sma3 = btind.SimpleMovingAverage(greater, period=self.p.period4)
        # self.params (shorthand: self.p)
        self.movav = btind.SimpleMovingAverage(self.data, period=self.p.period)
        # data0 is a daily data added from 'cerebro.adddata'
        sma4 = btind.SMA(self.data0, period=15)  # 15 days sma
        # data1 is a weekly data
        sma5 = btind.SMA(self.data1, period=5)
        self.buysig = sma4 > sma5()
        # Once again a potential implementation of a SimpleMovingAverage, further broken down into steps.
        # Sum N period values - datasum is now a *Lines* object
        # that when queried with the operator [] and index 0
        # returns the current sum
        datasum = btind.SumN(self.data, period=self.params.period)
        # datasum (being *Lines* object although single line) can be
        # naturally divided by an int/float as in this case. It could
        # actually be divided by anothr *Lines* object.
        # The operation returns an object assigned to "av" which again
        # returns the current average at the current instant in time
        # when queried with [0]
        av = datasum / self.params.period
        # The av *Lines* object can be naturally assigned to the named
        # line this indicator delivers. Other objects using this
        # indicator will have direct access to the calculation
        self.lines.sma = av
        sma6 = btind.SimpleMovingAverage(self.data, period=20)
        sma7 = btind.SimpleMovingAverage(self.data, period=15)
        close_over_sma = self.data.close > sma6
        sma_dist_to_high = self.data.high - sma6

        sma_dist_small = sma_dist_to_high < 3.5
        # Unfortunately "and" cannot be overridden in Python being
        # a language construct and not an operator and thus a
        # function has to be provided by the platform to emulate it
        sell_sig = bt.And(close_over_sma, sma_dist_small)
        self.buysig = bt.And(sma7 > self.data.close, sma7 > self.data.high)

        # bt.If the value of the sma is larger than close, return low, else return high
        high_or_low = bt.If(sma1 > self.data.close, self.data.low, self.data.high)
        sma8 = btind.SMA(high_or_low, period=15)

    def next(self):
        if self.movav.lines.sma[0] > self.data.lines.close[0]:
            # also valid: if self.movav.lines.sma > self.data.lines.close:
            print('Simple Moving Average is greater than the closing price')
        # self.data It has a lines attribute which contains a close attribute in turn
        # self.movav which is a SimpleMovingAverage indicator It has a lines attribute which contains a sma attribute in turn
        # shorthand for accessing line:
        # xxx.lines can be shortened to xxx.l
        # xxx.lines.name can be shortened to xxx.lines_name
        if self.data.close[0] > 30.0:
            # also valid: if self.data.lines.close[0] > 30.0:
            ...
        # A SimpleMovingAverage can be calculated for the current get/set point as follows:
        self.line[0] = math.fsum(self.data.get(0, size=self.p.period)) / self.p.period
        # the current close to the previous close is a 0 vs -1 thing.
        if self.data.close[0] > self.data.close[-1]:
            print('Closing price is higher today')
        if self.buysig[0]:
            print('daily sma is greater than weekly sma1')
        # Although this does not seem like an "operator" it actually is
        # in the sense that the object is being tested for a True/False
        # response
        if self.sma6 > 30.0:
            print('sma is greater than 30.0')
        if self.sma6 > self.data.close:
            print('sma is above the close price')
        # if self.sma > 30.0: … compares self.sma[0] to 30.0 (1st line and current value)
        # if self.sma > self.data.close: … compares self.sma[0] to self.data.close[0]
        if self.sell_sig:
            print('sell sig is True')
        else:
            print('sell sig is False')
        if self.sma_dist_to_high > 5.0:
            print('distance from sma to hig is greater than 5.0')
        if self.buysig[0]:
            pass  # do something


class SimpleMovingAverage(bt.Indicator):
    lines = ('sma',)
    # self.lines[0] points to self.lines.sma
    # shorthand
    # self.line points to self.lines[0]
    # self.lineX point to self.lines[X]
    # self.line_X point to self.lines[X]
    # self.dataY points to self.data.lines[Y]
    # self.dataX_Y points to self.dataX.lines[X] which is a full shorthard version of self.datas[X].lines[Y]

