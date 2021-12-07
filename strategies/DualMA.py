import backtrader as bt
import arrow

# 均线交叉策略
class DualMA(bt.Strategy):
    params = (('nfast', 10), ('nslow', 30), ('fgPrint', False),)

    # def log(self, txt, dt=None, fgPrint=False):
    #     ''' Logging function fot this strategy'''
    #     if self.params.fgPrint or fgPrint:
    #         dt = dt or self.datas[0].datetime.date(0)
    #         tn = tk.timNSec('', self.tim0wrk)
    #         # print('%s, %s，tn：%.2f' % (dt.isoformat(), txt))
    #         print('%s, %s，tim：%.2f s' % (dt.isoformat(), txt, tn))

    def __init__(self, vdict={}):
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.tim0wrk = arrow.now()
        #
        # self.dataclose = self.datas[0].close
        # print('@@vdict',vdict)
        #
        if len(vdict) > 0:
            self.p.nfast = int(vdict.get('nfast'))
            self.p.nslow = int(vdict.get('nslow'))

        #
        sma_fast, sma_slow, self.buysig = {}, {}, {}
        for xc, xdat in enumerate(self.datas):
            sma_fast[xc] = bt.ind.SMA(xdat, period=self.p.nfast)
            # sma_fast[xc] =bt.ema.SMA(xdat,period=self.p.nfast)
            sma_slow[xc] = bt.ind.SMA(xdat, period=self.p.nslow)
            self.buysig[xc] = bt.ind.CrossOver(sma_fast[xc], sma_slow[xc])
        #

    def next(self):
        # self.log('Close, %.2f' % self.dataclose[0])
        #
        if self.order:
            return

        #
        for xc, xdat in enumerate(self.datas):
            xnam = xdat._name
            fgInx = xnam.find('inx_') >= 0
            # print('@n',xnam,fgInx)
            if not fgInx:
                xpos = self.getposition(xdat)
                # xnam=xdat.line.name
                xss = ' {:.02},@ : {} # {}'.format(xdat.close[0], xnam, xc)
                if xpos.size:
                    # if (self.buysig[xc] < 0)and(self.buysig[0] < 0):
                    if (self.buysig[xc] < 0):
                        self.log('SELL' + xss)
                        # self.log('    @@SELL CREATE, %.2f, %'  % (self.datas[xc].close[0],self.data0_Name))
                        self.sell(data=xdat)

                # elif (self.buysig[xc] > 0)and(self.buysig[0] > 0):
                elif (self.buysig[xc] > 0):
                    self.log('BUY' + xss)
                    # self.log('    @@buy CREATE, %.2f'  % (self.datas[xc].close[0]))
                    # self.log('    @@buy CREATE, %.2f, %'  % (self.datas[xc].close[0],self.datas[xc].Name))
                    self.buy(data=xdat)
                    #
                    # self.order = self.sell(data=xdat,exectype=bt.Order.StopTrail,trailpercent=0.0618)

    def stop(self):
        # tn=arrow.now()-self.tim0wrk
        self.log('策略参数 nfast= %2d, nslow= %2d，资产总值：%.2f' %
                 (self.p.nfast, self.p.nslow, self.broker.getvalue()), fgPrint=True)
