import backtrader as bt
from datetime import datetime
import time


class TrendLine(bt.Indicator):
    '''
    This indicator shall produce a signal when price reaches a calculated trend line.

    The indicator requires two price points and date points to serve as X and Y
    values in calcuating the slope and the future expected price trend

    x1 = Date/Time, String in the following format "YYYY-MM-DD HH:MM:SS" of
    the start of the trend
    y1 = Float, the price (Y value) of the start of the trend.
    x2 = Date/Time, String in the following format "YYYY-MM-DD HH:MM:SS" of
    the end of the trend
    y2 = Float, the price (Y value) of the end of the trend.
    '''

    lines = ('signal', 'trend')
    params = (
        ('x1', None),
        ('y1', None),
        ('x2', None),
        ('y2', None)
    )

    def __init__(self):
        self.p.x1 = datetime.datetime.strptime(self.p.x1, "%Y-%m-%d %H:%M:%S")
        self.p.x2 = datetime.datetime.strptime(self.p.x2, "%Y-%m-%d %H:%M:%S")
        x1_time_stamp = time.mktime(self.p.x1.timetuple())
        x2_time_stamp = time.mktime(self.p.x2.timetuple())
        self.m = self.get_slope(x1_time_stamp, x2_time_stamp, self.p.y1, self.p.y2)
        self.B = self.get_y_intercept(self.m, x1_time_stamp, self.p.y1)
        self.plotlines.trend._plotskip = True

    def next(self):
        date = self.data0.datetime.datetime()
        date_timestamp = time.mktime(date.timetuple())
        Y = self.get_y(date_timestamp)
        self.lines.trend[0] = Y

        # Check if price has crossed up / down into it.
        if self.data0.high[-1] < Y and self.data0.high[0] > Y:
            self.lines.signal[0] = -1
            return

        # Check for cross downs (Into support)
        elif self.data0.low[-1] > Y and self.data0.low[0] < Y:
            self.lines.signal[0] = 1
            return

        else:
            self.lines.signal[0] = 0

    def get_slope(self, x1, x2, y1, y2):
        m = (y2 - y1) / (x2 - x1)
        return m

    def get_y_intercept(self, m, x1, y1):
        b = y1 - m * x1
        return b

    def get_y(self, ts):
        Y = self.m * ts + self.B
        return Y
