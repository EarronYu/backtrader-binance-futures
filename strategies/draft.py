from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt


class BBSqueeze(bt.Indicator):
    """
    https://www.netpicks.com/squeeze-out-the-chop/
    Both indicators are symmetrical, meaning that the upper and lower bands or channel lines are the same distance from the moving average. That means that
    we can focus on only one side in developing our indicator. In our case, we’ll just consider the upper lines.
    The basic formulas we need are:
        Bollinger Band = Moving Average + (Number of standard deviations X Standard Deviation)
        Keltner Channel = Moving Average + (Number of ATR’s X ATR)
    Or if we translate this into pseudo-code:
        BBUpper = Avg(close, period) + (BBDevs X StdDev(close, period))
        KCUpper = Avg(close, period) + (KCDevs X ATR(period))
    The squeeze is calculated by taking the difference between these two values:
        Squeeze = BBUpper – KCUpper
    Which simplifies down to this:
        Squeeze = (BBDevs X StdDev(close, period)) – (KCDevs X ATR(period))
    """

    lines = ('squeeze',)
    params = (('period', 20), ('bbdevs', 2.0), ('kcdevs', 1.5), ('movav', bt.ind.MovAv.Simple),)
    plotinfo = dict(subplot=True)

    def _plotlabel(self):
        plabels = [self.p.period, self.p.bbdevs, self.p.kcdevs]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def __init__(self):
        bb = bt.ind.BollingerBands(
            period=self.p.period, devfactor=self.p.bbdevs, movav=self.p.movav)
        kc = KeltnerChannel(
            period=self.p.period, devfactor=self.p.kcdevs, movav=self.p.movav)
        self.lines.squeeze = bb.top - kc.top


class KeltnerChannel(bt.Indicator):

    lines = ('mb', 'top', 'bot',)
    params = (('period', 20), ('devfactor', 1.5),
              ('movav', bt.ind.MovAv.Simple),)

    plotinfo = dict(subplot=False)
    plotlines = dict(
        mid=dict(ls='--'),
        top=dict(_samecolor=True),
        bot=dict(_samecolor=True),
    )

    def _plotlabel(self):
        plabels = [self.p.period, self.p.devfactor]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def __init__(self):
        self.lines.mid = ma = self.p.movav(self.data, period=self.p.period)
        atr = self.p.devfactor * bt.ind.ATR(self.data, period=self.p.period)
        self.lines.top = ma + atr
        self.lines.bot = ma - atr