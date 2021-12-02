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

'''
- Currently (last two period is training set for next period, but I want the total to be validated on every single period) --> switch!
- strategy should be easy to replace
- implement daily vs. weekly etc. compression)
- improve the sizer (depending on the strategy)
- Implement number of trades (greater than 50 + # var * 50 is suggested by the author (I assume this is meant for daily values) & volatility of trade returns for testing and training to indicate the statistical fitness of the model (https://www.tradelikeamachine.com/blog/over-optimization/part-3-in-sample-out-of-sample-trading-system-backtesting integrate )(volatility of trades?) --> self.returnvola = btind.StdDev(btind.PctChange(), period=30)
https://www.tradelikeamachine.com/images/blog/over-optimization/quantitative-out-of-sample-walk-forward-metrics.png
- pyfolio integration
- better charting?
- btreport?
- clean up code?
- find a momentum strategy to implement
- etc.
'''
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

globalparams = dict(strategy='long_short',  # if a different strategy is used
                    n_splits=10,  # how many chunks the data should have
                    fixed_length=True,
                    # by setting it False the training data will will grow over time, otherwise it will keep the size given under train_splits
                    train_splits=1,
                    # how many splits should be used to train the model (be aware of these two variables may not work with your strategy, i.e. SMA100 but two training splits are just 110 days or less than 100 days)
                    test_splits=1,  # how many splits should the data be tested on?
                    start=dt.datetime(2010, 1, 1),
                    end=dt.datetime(2020, 1, 31),
                    symbols=["TQQQ"],  # , "GOOG", "MSFT", "AMZN", "SNY", "VZ", "IBM", "HPQ", "QCOM", "NVDA"
                    cash=10000,
                    commission=0.02,
                    coc='True',
                    num_evals=100,  # how often should the optimizer try to optimize
                    var1range=[1, 1.25],  # reasonable range within the optimization should happen (variable 1)
                    var2range=[1, 1.25],  # reasonable range within the optimization should happen (variable 2)
                    sma_period=15,  # SMA Band Period
                    vola=False,
                    # this should only be used if one has a working strategy: if True the total period will be optimized and then the volatility of the overall parameters can be observed todo implement it
                    )


class TimeSeriesSplitImproved(TimeSeriesSplit):
    """Time Series cross-validator
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide `.
    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.
    Examples
    --------
    # >>> from sklearn.model_selection import TimeSeriesSplit
    # >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    # >>> y = np.array([1, 2, 3, 4])
    # >>> tscv = TimeSeriesSplit(n_splits=3)
    # >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(n_splits=3)
    # >>> for train_index, test_index in tscv.split(X):
    # ...    print("TRAIN:", train_index, "TEST:", test_index)
    # ...    X_train, X_test = X[train_index], X[test_index]
    # ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    # >>> for train_index, test_index in tscv.split(X, fixed_length=True):
    # ...     print("TRAIN:", train_index, "TEST:", test_index)
    # ...     X_train, X_test = X[train_index], X[test_index]
    # ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [1] TEST: [2]
    TRAIN: [2] TEST: [3]
    # >>> for train_index, test_index in tscv.split(X, fixed_length=True,
    # ...     train_splits=2):
    # ...     print("TRAIN:", train_index, "TEST:", test_index)
    # ...     X_train, X_test = X[train_index], X[test_index]
    # ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [1 2] TEST: [3]
    Notes
    -----
    When ``fixed_length`` is ``False``, the training set has size
    ``i * train_splits * n_samples // (n_splits + 1) + n_samples %
    (n_splits + 1)`` in the ``i``th split, with a test set of size
    ``n_samples//(n_splits + 1) * test_splits``, where ``n_samples``
    is the number of samples. If fixed_length is True, replace ``i``
    in the above formulation with 1, and ignore ``n_samples %
    (n_splits + 1)`` except for the first training set. The number
    of test sets is ``n_splits + 2 - train_splits - test_splits``.
    """

    def split(self, X, y=None, groups=None, fixed_length=False,
              train_splits=1, test_splits=1):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        fixed_length : bool, hether training sets should always have
            common length
        train_splits : positive int, for the minimum number of
            splits to include in training sets
        test_splits : positive int, for the number of splits to
            include in the test set
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if ((n_folds - train_splits - test_splits) == 0 and test_splits > 0):
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)

        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])


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


class SMAC(bt.Strategy):
    """A simple moving average crossover strategy; crossing of a fast and slow moving average generates buy/sell
       signals"""
    params = dict(var1=20, var2=50)  # The windows for both var1 (fast) and var2 (slow) moving averages

    def __init__(self):
        """Initialize the strategy"""

        self.fastma = dict()
        self.slowma = dict()
        self.regime = dict()

        if self.params.var1 > self.params.var2:
            vfast = self.params.var1
            self.params.var1 = self.params.var2
            self.params.var2 = vfast

        for d in self.getdatanames():
            # The moving averages
            self.fastma[d] = btind.SMA(self.getdatabyname(d),  # The symbol for the moving average
                                       period=self.params.var1,  # Fast moving average
                                       plotname="FastMA: " + d)
            self.slowma[d] = btind.SMA(self.getdatabyname(d),  # The symbol for the moving average
                                       period=self.params.var2,  # Slow moving average
                                       plotname="SlowMA: " + d)

            # Get the regime
            self.regime[d] = self.fastma[d] - self.slowma[d]  # Positive when bullish

    def next(self):
        """Define what will be done in a single step, including creating and closing trades"""
        for d in self.getdatanames():  # Looping through all symbols
            pos = self.getpositionbyname(d).size or 0
            if pos == 0:  # Are we out of the market?
                # Consider the possibility of entrance
                # Notice the indexing; [0] always means the present bar, and [-1] the bar immediately preceding
                # Thus, the condition below translates to: "If today the regime is bullish (greater than
                # 0) and yesterday the regime was not bullish"
                if self.regime[d][0] > 0 and self.regime[d][-1] <= 0:  # A buy signal
                    self.buy(data=self.getdatabyname(d))

            else:  # We have an open position
                if self.regime[d][0] <= 0 and self.regime[d][-1] > 0:  # A sell signal
                    self.sell(data=self.getdatabyname(d))


class PropSizer(bt.Sizer):  # todo need rework
    """A position sizer that will buy as many stocks as necessary for a certain proportion of the portfolio
       to be committed to the position, while allowing stocks to be bought in batches (say, 100)"""
    params = {"prop": 0.95, "batch": 100}

    def _getsizing(self, comminfo, cash, data, isbuy):
        """Returns the proper sizing"""

        if isbuy:  # Buying
            target = self.broker.getvalue() * self.params.prop  # Ideal total value of the position
            price = data.close[0]
            shares_ideal = target / price  # How many shares are needed to get target
            batches = int(shares_ideal / self.params.batch)  # How many batches is this trade?
            shares = batches * self.params.batch  # The actual number of shares bought

            if shares * price > cash:
                return 0  # Not enough money for this trade ????????? Why not buy anything if there is no money?
            else:
                return shares

        else:  # Selling
            return self.broker.getposition(data).size  # Clear the position


class AcctValue(bt.Observer):
    alias = ('Value',)
    lines = ('value',)

    plotinfo = {"plot": True, "subplot": True}

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()  # Get today's account value (cash + stocks)


class AcctStats(bt.Analyzer):
    """A simple analyzer that gets the gain in the value of the account; should be self-explanatory"""

    def __init__(self):
        self.start_val = self.strategy.broker.get_value()
        self.end_val = None

    def stop(self):
        self.end_val = self.strategy.broker.get_value()

    def get_analysis(self):
        return {"start": self.start_val, "end": self.end_val,
                "growth": self.end_val - self.start_val, "return": self.end_val / self.start_val}


# Different stocks from past posts because of different data source (no plot for NTDOY)

datafeeds = {s: web.DataReader(s, "yahoo", globalparams['start'], globalparams['end']) for s in globalparams['symbols']}

# resample to weekly data

for s, df in datafeeds.items():
    '''
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    '''


    def take_first(array_like):
        return array_like[0]


    def take_last(array_like):
        return array_like[-1]


    ohlc_dict = {'High': 'max',
                 'Low': 'min',
                 'Open': take_first,
                 'Close': take_last,
                 'Adj Close': take_last,
                 'Volume': 'sum'}

    # datafeeds[s] = df.resample('W',  # Weekly resample
    #                            loffset=pd.offsets.timedelta(days=-2)).agg(
    #     ohlc_dict).copy()  # to put the labels to Monday
    datafeeds[s] = df.resample('W',  # Weekly resample
                               offset='-2d').agg(
        ohlc_dict).copy()  # to put the labels to Monday
for df in datafeeds.values():
    df["OpenInterest"] = 0  # PandasData reader expects an OpenInterest column;

tscv = TimeSeriesSplitImproved(globalparams['n_splits'])
split = tscv.split(datafeeds[globalparams['symbols'][0]], fixed_length=globalparams['fixed_length'],
                   train_splits=globalparams['train_splits'], test_splits=globalparams['test_splits'])

walk_forward_results = list()

# Be prepared: this will take a while
for train, test in split:

    # TRAINING

    # Optimize with optunity
    def runstrat(var1, var2):
        cerebro = bt.Cerebro(stdstats=False, maxcpus=None)
        cerebro.addstrategy(eval(globalparams['strategy']), var1=var1,
                            var2=var2)  # toDO make the float int choice switchable
        cerebro.broker.setcash(globalparams['cash'])
        cerebro.broker.setcommission(globalparams['commission'])
        for s, df in datafeeds.items():
            data = bt.feeds.PandasData(dataname=df.iloc[train], name=s)  # Add a subset of data
            # to the object that
            # corresponds to training
            cerebro.adddata(data)
        cerebro.broker.set_coc(eval(globalparams['coc']))
        cerebro.run()
        return cerebro.broker.getvalue()  # ToDo make the variable that should be optimized flexible


    opt = optunity.maximize(runstrat, num_evals=globalparams['num_evals'], var1=globalparams['var1range'],
                            var2=globalparams['var2range'])

    optimal_pars, details, _ = opt

    tester = bt.Cerebro(stdstats=False, maxcpus=None)
    tester.broker.set_cash(globalparams['cash'])
    tester.broker.set_coc(eval(globalparams['coc']))
    tester.broker.setcommission(globalparams['commission'])
    tester.addanalyzer(AcctStats)
    tester.addsizer(PropSizer)
    tester.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)

    # TESTING
    tester.addstrategy(eval(globalparams['strategy']), var1=optimal_pars['var1'],
                       var2=optimal_pars['var2'])  # Test with optimal combination toDO like above int vs float
    for s, df in datafeeds.items():
        data = bt.feeds.PandasData(dataname=df.iloc[test], name=s)  # Add a subset of data
        # to the object that
        # corresponds to testing
        tester.adddata(data)

    res = tester.run()
    res_dict = res[0].analyzers.acctstats.get_analysis()
    res_dict["var1"] = optimal_pars['var1']
    res_dict["var2"] = optimal_pars['var2']
    res_dict["sharpe"] = res[0].analyzers.sharperatio.get_analysis()['sharperatio']
    res_dict["start_date"] = datafeeds[globalparams['symbols'][0]].iloc[test[0]].name
    res_dict["end_date"] = datafeeds[globalparams['symbols'][0]].iloc[test[-1]].name
    walk_forward_results.append(res_dict)

wfdf = DataFrame(walk_forward_results)
print(wfdf.loc[:, wfdf.columns != 'start'])


class SMACWalkForward(bt.Strategy):
    """The SMAC strategy but in a walk-forward analysis context"""
    params = {"start_dates": None,  # Starting days for trading periods (a list)
              "end_dates": None,  # Ending day for trading periods (a list)
              "var1": None,  # List of fast moving average windows, corresponding to start dates (a list)
              "var2": None}  # Like fast, but for slow moving average window (a list)

    # All the above lists must be of the same length, and they all line up

    def __init__(self):
        """Initialize the strategy"""
        self.sma = dict()
        self.var1 = dict()
        self.var2 = dict()
        self.regime = dict()

        self.date_combos = [c for c in zip(self.p.start_dates, self.p.end_dates)]
        # Error checking
        if type(self.p.start_dates) is not list or type(self.p.end_dates) is not list or \
                type(self.p.fast) is not list or type(self.p.slow) is not list:
            raise ValueError("Must past lists filled with numbers to params start_dates, end_dates, fast, slow.")
        elif len(self.p.start_dates) != len(self.p.end_dates) or \
                len(self.p.fast) != len(self.p.start_dates) or len(self.p.slow) != len(self.p.start_dates):
            raise ValueError("All lists passed to params must have same length.")
        for d in self.getdatanames():
            self.sma[d] = dict()
            self.var1[d] = dict()
            self.var2[d] = dict()
            self.regime[d] = dict()

            # Additional indexing, allowing for differing start/end dates
            for sd, ed, f, s in zip(self.p.start_dates, self.p.end_dates, self.p.var1, self.p.var2):
                # More error checking
                '''
                if type(f) is not int or type(s) is not int:
                    raise ValueError("Must include only integers in fast, slow.")
                elif f > s:
                    raise ValueError("Elements in fast cannot exceed elements in slow.")
                elif f <= 0 or s <= 0:
                    raise ValueError("Moving average windows must be positive.")


                if type(sd) is not dt.date or type(ed) is not dt.date:
                    raise ValueError("Only datetime dates allowed in start_dates, end_dates.")
                elif ed - sd < dt.timedelta(0):
                    raise ValueError("Start dates must always be before end dates.")
                '''
                # The moving averages
                # Notice that different moving averages are obtained for different combinations of
                # start/end dates
                self.sma[d][(sd, ed)] = btind.SimpleMovingAverage(self.getdatabyname(d),
                                                                  period=globalparams['sma_period'],
                                                                  plot=False)
                self.var1[d][(sd, ed)] = f
                self.var2[d][(sd, ed)] = s
                '''
                self.fastma[d][(sd, ed)] = btind.SimpleMovingAverage(self.getdatabyname(d),
                                                                     period=f,
                                                                     plot=False)
                self.slowma[d][(sd, ed)] = btind.SimpleMovingAverage(self.getdatabyname(d),
                                                                     period=s,
                                                                     plot=False)

                # Get the regime
                self.regime[d][(sd, ed)] = self.fastma[d][(sd, ed)] - self.slowma[d][(sd, ed)]
                # In the future, use the backtrader indicator btind.CrossOver()
                '''

    def next(self):
        """Define what will be done in a single step, including creating and closing trades"""

        # Determine which set of moving averages to use
        curdate = self.datetime.date(0)
        dtidx = None  # Will be index
        # Determine which period (if any) we are in
        for sd, ed in self.date_combos:
            # Debug output
            # print('{}: {} < {}: {}, {} < {}: {}'.format(
            #    len(self), sd, curdate, (sd <= curdate), curdate, ed, (curdate <= ed)))
            if sd <= curdate and curdate <= ed:
                dtidx = (sd, ed)
        # Debug output
        # print('{}: the dtixdx is {}, and curdate is {};'.format(len(self), dtidx, curdate))
        for d in self.getdatanames():  # Looping through all symbols
            pos = self.getpositionbyname(d).size or 0
            if dtidx is None:  # Not in any window
                break  # Don't engage in trades
            if pos == 0:  # Are we out of the market?
                # Consider the possibility of entrance
                # Notice the indexing; [0] always mens the present bar, and [-1] the bar immediately preceding
                # Thus, the condition below translates to: "If today the regime is bullish (greater than
                # 0) and yesterday the regime was not bullish"
                '''if self.slowma[d][dtidx][0] > self.getdatabyname(d).close[0]:  # A buy signal
                    self.sell(data=self.getdatabyname(d), size=1000)

            else:  # We have an open position
                if self.fastma[d][dtidx][0] < self.getdatabyname(d).close[0]:  # A sell signal
                    self.close(data=self.getdatabyname(d), size=1000)
            '''
                if self.sma[d][dtidx][0] * self.var1[d][dtidx] > self.getdatabyname(d).high[0]:  # A buy signal
                    self.order_target_percent(data=self.getdatabyname(d), target=0.98)

            else:  # We have an open position
                if self.getdatabyname(d).close[-1] * self.var2[d][dtidx] <= self.getdatabyname(d).high[0]:  # A sell signal
                    self.order_target_percent(data=self.getdatabyname(d), target=0)


cerebro_wf = bt.Cerebro(stdstats=False, maxcpus=None)

plot_symbols = globalparams['symbols']
is_first = True

# plot_symbols = []
for s, df in datafeeds.items():
    data = bt.feeds.PandasData(dataname=df, name=s)
    if s in plot_symbols:
        if is_first:
            data_main_plot = data
            is_first = False
        else:
            data.plotinfo.plotmaster = data_main_plot
    else:
        data.plotinfo.plot = True
    cerebro_wf.adddata(data)  # Give the data to cerebro

cerebro_wf.broker.setcash(globalparams['cash'])
cerebro_wf.broker.setcommission(globalparams['commission'])
cerebro_wf.broker.set_coc(eval(globalparams['coc']))

cerebro_wf.addstrategy(SMACWalkForward,
                       # Give the results of the above optimization to SMACWalkForward (NOT OPTIONAL)
                       var1=[f for f in wfdf.var1],
                       var2=[s for s in wfdf.var2],
                       start_dates=[sd.date() for sd in wfdf.start_date],
                       end_dates=[ed.date() for ed in wfdf.end_date])
cerebro_wf.addobserver(AcctValue)
cerebro_wf.addobservermulti(bt.observers.BuySell)  # Plots up/down arrows
cerebro_wf.addsizer(PropSizer)
cerebro_wf.addanalyzer(AcctStats)
cerebro_wf.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)

results = cerebro_wf.run()
print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")

cerebro_wf.plot(iplot=True)
