import backtrader as bt
import backtrader.indicators as btind
import datetime as dt
import pandas as pd
import pandas_datareader as web
from pandas import Series, DataFrame
import random
from copy import deepcopy


class SMAC(bt.Strategy):
    """A simple moving average crossover strategy; crossing of a fast and slow moving average generates buy/sell
       signals"""
    params = {"fast": 20, "slow": 50,  # The windows for both fast and slow moving averages
              "optim": False, "optim_fs": (20, 50)}  # Used for optimization; equivalent of fast and slow, but a tuple

    # The first number in the tuple is the fast MA's window, the
    # second the slow MA's window

    def __init__(self):
        """Initialize the strategy"""

        self.fastma = dict()
        self.slowma = dict()
        self.regime = dict()

        if self.params.optim:  # Use a tuple during optimization
            self.params.fast, self.params.slow = self.params.optim_fs  # fast and slow replaced by tuple's contents

        if self.params.fast > self.params.slow:
            raise ValueError(
                "A SMAC strategy cannot have the fast moving average's window be " + \
                "greater than the slow moving average window.")

        for d in self.getdatanames():
            # The moving averages
            self.fastma[d] = btind.SimpleMovingAverage(self.getdatabyname(d),  # The symbol for the moving average
                                                       period=self.params.fast,  # Fast moving average
                                                       plotname="FastMA: " + d)
            self.slowma[d] = btind.SimpleMovingAverage(self.getdatabyname(d),  # The symbol for the moving average
                                                       period=self.params.slow,  # Slow moving average
                                                       plotname="SlowMA: " + d)

            # Get the regime
            self.regime[d] = self.fastma[d] - self.slowma[d]  # Positive when bullish

    def next(self):
        """Define what will be done in a single step, including creating and closing trades"""
        for d in self.getdatanames():  # Looping through all symbols
            pos = self.getpositionbyname(d).size or 0
            if pos == 0:  # Are we out of the market?
                # Consider the possibility of entrance
                # Notice the indexing; [0] always mens the present bar, and [-1] the bar immediately preceding
                # Thus, the condition below translates to: "If today the regime is bullish (greater than
                # 0) and yesterday the regime was not bullish"
                if self.regime[d][0] > 0 and self.regime[d][-1] <= 0:  # A buy signal
                    self.buy(data=self.getdatabyname(d))

            else:  # We have an open position
                if self.regime[d][0] <= 0 and self.regime[d][-1] > 0:  # A sell signal
                    self.sell(data=self.getdatabyname(d))


class PropSizer(bt.Sizer):
    """A position sizer that will buy as many stocks as necessary for a certain proportion of the portfolio
       to be committed to the position, while allowing stocks to be bought in batches (say, 100)"""
    params = {"prop": 0.1, "batch": 100}

    def _getsizing(self, comminfo, cash, data, isbuy):
        """Returns the proper sizing"""

        if isbuy:  # Buying
            target = self.broker.getvalue() * self.params.prop  # Ideal total value of the position
            price = data.close[0]
            shares_ideal = target / price  # How many shares are needed to get target
            batches = int(shares_ideal / self.params.batch)  # How many batches is this trade?
            shares = batches * self.params.batch  # The actual number of shares bought

            if shares * price > cash:
                return 0  # Not enough money for this trade
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


def createWFAReport(simParams, simulations):
    # Create simulation report format
    reportColumns = ['grossProfit', 'grossAverageProfit', 'maxProfit',
                     'grossLoss', 'grossAverageLoss', 'maxLoss',
                     'netProfit', 'averageNetProfit', 'NAR',
                     'recoveryFactor', 'MDDLength', 'MDD',
                     'wonTrade', 'lossTrade', 'tradingTime',
                     'averageTradeTime', 'TradeNumber', 'maxValue',
                     'minValue', 'totalCommission']
    simulationReport = pd.DataFrame(columns=reportColumns)

    # Loop Simulations to create summary
    for simulation in simulations:
        '''some Calculation is done here'''
    return simReport


def WFASplit(self, trainBy='12m', testBy='3m', loopBy='m', overlap=True):
    startDate = self.index[0]
    endDate = self.index[-1]
    if trainBy[-1] is 'm':
        trainTime = relativedelta(months=int(trainBy[:-1]))
    else:
        raise ValueError
    if testBy[-1] is 'm':
        testTime = relativedelta(months=int(testBy[:-1]))
    else:
        raise ValueError
    assert ((relativedelta(endDate, startDate) - trainTime).days) > 0

    if loopBy is 'm':
        test_starts = zip(rrule(MONTHLY, dtstart=startDate, until=endDate - trainTime, interval=int(testBy[:-1])))
    else:
        raise ValueError

    for i in test_starts:
        startD = i[0]
        endD = i[0] + trainTime
        yield (self[(self.index >= startD) & (self.index < endD)],
               self[(self.index >= endD) & (self.index < endD + testTime)])
    return None


def runTrain(trainTestGenerator, _ind, stockName):
    WFATrainResult = []
    for train, test in trainTestGenerator:
        logger.debug(
            '{} Training Data:{} to {}'.format(stockName, pd.DatetimeIndex.strftime(train.head(1).index, '%Y-%m-%d'),
                                               pd.DatetimeIndex.strftime(train.tail(1).index, '%Y-%m-%d')))
        # Generate Indicator ResultSet
        trainer = bt.Cerebro(cheat_on_open=True, stdstats=False, optreturn=False)
        trainer.broker.set_cash(10000)
        # Add Commission
        IB = params['commission'](commission=0.0)
        trainer.broker.addcommissioninfo(IB)
        # Below Analyzer are used to calculate the Recovery Ratio
        trainer.addanalyzer(btanalyzers.TradeAnalyzer, _name='TradeAn')
        trainer.addanalyzer(recoveryAnalyzer, timeframe=params['analysisTimeframe'], _name='recoveryFac')
        trainer.addanalyzer(WFAAn, _name='WFAAna')
        trainer.addanalyzer(btanalyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='TR')
        # SetBroker
        trainer.broker.set_checksubmit(False)
        # Copy for tester
        tester = deepcopy(trainer)
        # Optimize Strategy
        trainingFile = '{}/WFA'
        trainer.optstrategy(trainingIdea,
                            inOrOut=(params['inOrOut'],),
                            selfLog=(params['selfLog'],),
                            indName=(row.indicator,),
                            indFormula=(_ind['formula'],),
                            entryExitPara=(_ind['entryExitParameters'],),
                            indOutName=(_ind['indValue'],),
                            nonOptParams=(None,),
                            resultLocation=(params['resultLocation'],),
                            timeString=(params['timeString'],),
                            market=(row.market,),
                            **optt)
        trainData = bt.feeds.PandasData(dataname=train)
        # Add a subset of data.
        trainer.adddata(trainData)
        optTable = trainer.run()
        final_results_list = []
        for run in optTable:
            for x in run:
                x.params['res'] = x.analyzers.WFAAna.get_analysis()
                final_results_list.append(x.params)

        _bestWFA = \
        pd.DataFrame.from_dict(final_results_list, orient='columns').sort_values('res', ascending=False).iloc[
            0].to_dict()
        bestTrainParams = {key: _bestWFA[key] for key in _bestWFA if
                           key not in ['market', 'inOrOut', 'resultLocation', 'selfLog', 'timeString', 'res']}
        bestTrainParams = pd.DataFrame(bestTrainParams, index=[0])
        bestTrainParams['trainStart'] = train.iloc[0].name
        bestTrainParams['trainEnd'] = train.iloc[-1].name
        bestTrainParams['testStart'] = test.iloc[0].name
        bestTrainParams['testEnd'] = test.iloc[-1].name
        WFATrainResult.append(bestTrainParams)
    WFATrainResult = pd.concat(WFATrainResult)
    return WFATrainResult


def runTest(params, WFATrainResult, _ind, datafeed, stockName):
    # Generate Indicator ResultSet
    tester = bt.Cerebro(cheat_on_open=True)
    tester.broker.set_cash(10000)
    # Add Commission
    IB = params['commission'](commission=0.0)
    tester.broker.addcommissioninfo(IB)
    # SetBroker
    tester.broker.set_checksubmit(False)
    logger.debug('{} Start Testing'.format(stockName))
    OneSimHandler = logging.FileHandler(
        filename='{}/simulation/{}_{}_test.log'.format(params['resultLocation'], str(stockName), str(row.indicator)))
    OneSimHandler.setLevel(logging.DEBUG)
    OneSimHandler.setFormatter(logging.Formatter("%(asctime)s:%(relativeCreated)d - %(message)s"))
    oneLogger.addHandler(OneSimHandler)
    tester.addstrategy(trainingIdea,
                       inOrOut=params['inOrOut'],
                       selfLog=params['selfLog'],
                       indName=row.indicator,
                       indFormula=_ind['formula'],
                       entryExitPara=_ind['entryExitParameters'],
                       indOutName=_ind['indValue'],
                       nonOptParams=None,
                       resultLocation=params['resultLocation'],
                       timeString=params['timeString'],
                       market=market,
                       WFATestParams=WFATrainResult)
    data = bt.feeds.PandasData(dataname=datafeed)
    tester.adddata(data, name=stockName)
    # Add analyzers for Tester
    tester.addanalyzer(btanalyzers.DrawDown, _name='MDD')
    tester.addanalyzer(btanalyzers.TradeAnalyzer, _name='TradeAn')
    tester.addanalyzer(btanalyzers.SQN, _name='SQN')
    tester.addanalyzer(recoveryAnalyzer, timeframe=params['analysisTimeframe'], _name='recoveryFac')
    tester.addanalyzer(ITDDAnalyzer, _name='ITDD')
    tester.addanalyzer(simpleAn, _name='simpleAna')
    tester.addanalyzer(btanalyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='TR')
    # Run and Return Cerebro
    cere = tester.run()[0]

    _report = cere.analyzers.simpleAna.writeAnalysis(bnhReturn)
    oneLogger.removeHandler(OneSimHandler)
    if params['plotGraph']:
        plotSimGraph(tester, params, stockName, row.indicator)
    return _report


if __name__ == "__main__":
    session = 'WinsWFAProd'
    stockList, indicatorDict, params = jsonConfigMap(session)
    params['timeString'] = '0621_113833'
    params['topResult'] = pd.read_csv('{}{}/consoReport.csv'.format(params['resultLocation'], params['timeString']),
                                      index_col=0)
    params['resultLocation'] += params['timeString'] + '/WFA'
    simulations = []

    try:
        # Create Folder
        shutil.rmtree(params['resultLocation'])
    except FileNotFoundError:
        pass
    except PermissionError:
        pass
    os.makedirs(params['resultLocation'], exist_ok=True)
    for element in ['order', 'trade', 'mr', 'ohlc', 'simulation', 'bestTrain', 'graph']:
        os.makedirs(params['resultLocation'] + '/' + element, exist_ok=True)
    # Create master Log
    handler = logging.FileHandler(filename='{}/Master.log'.format(params['resultLocation']))
    handler.setFormatter(logging.Formatter('%(asctime)s:%(name)s - %(levelname)s {}- %(message)s'))
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    # Create OptReport Log
    reportHandler = logging.FileHandler(filename='{}/optReport.csv'.format(params['resultLocation']))
    reportHandler.setFormatter(logging.Formatter('%(message)s'))
    reportHandler.setLevel(logging.INFO)
    reportLogger = logging.getLogger('report')
    reportLogger.addHandler(reportHandler)
    simResultColumns = ['stockName', 'market', 'indicator',
                        'grossProfit', 'grossAverageProfit', 'maxProfit',
                        'grossLoss', 'grossAverageLoss', 'maxLoss',
                        'netProfit', 'averageNetProfit', 'NAR', 'profitFactor',
                        'recoveryFactor', 'MDDLength', 'MDD',
                        'selfMDD', 'winRate', 'tradingTimeRatio',
                        'averageTradingBar', 'tradeNumber', 'maxValue',
                        'minValue', 'initialv', 'totalCommission',
                        'barNumber', 'expectancy100', 'bnhReturn', 'bnhRatio']
    reportLogger.info(str(simResultColumns).strip("[]").replace("'", "").replace(" ", ""))

    # Create Simulation Log
    oneLogger = logging.getLogger('oneLogger')
    oneLogger.propagate = False
    postHandler = logging.StreamHandler()
    postHandler.setLevel(logging.INFO)
    if params['selfLog']:
        oneLogger.setLevel(logging.DEBUG)
    else:
        oneLogger.setLevel(logging.INFO)
    oneLogger.addHandler(postHandler)
    # Record Start Time
    startTime = time.time()
    for row in params['topResult'].itertuples():
        simParams = pd.DataFrame(columns=['startTime', 'endTime', 'Parameter'])
        indicator = indicatorDict[row.indicator]
        stockName = row.stockName
        market = row.market
        try:
            optt = eval(indicator['optParam'])
        except:
            logger.info('{}: Indicator does not have WFA parameters and skipped'.format(row.indicator))
            continue
        datafeed = feeder(stockName, market, params)
        bnhReturn = round(datafeed.iloc[-1]['close'] - datafeed.iloc[0]['open'], 2)
        # Extract Feeder from Data to save time for multi-simulation
        print('Start WFA for {}-{} from {} to {}'.format(stockName, row.indicator, datafeed.iloc[0].name,
                                                         datafeed.iloc[-1].name))
        trainTestGenerator = WFASplit(datafeed, trainBy='8m', testBy='8m', loopBy='m')
        _ind = indicatorDict[row.indicator]
        # Training
        WFATrainResult = runTrain(trainTestGenerator, _ind, stockName)
        WFATrainResult = pd.DataFrame.from_records(WFATrainResult)
        WFATrainResult.to_csv('{}/bestTrain/{}_{}_train.csv'.format(params['resultLocation'], stockName, row.indicator),
                              index=False)
        # TESTING
        _report = runTest(params, WFATrainResult, _ind, datafeed, stockName)
        # Consolidate T
        if _report[0] is not None:
            reportLogger.info(str(_report).strip("[]").strip(" ").replace(" ", "").replace("'", ""))
            simulations.append(_report)

    # After simulations
    simulations = pd.DataFrame(simulations)
    reportColumns = ['stockName', 'market', 'indicator',
                     'grossProfit', 'grossAverageProfit', 'maxProfit',
                     'grossLoss', 'grossAverageLoss', 'maxLoss',
                     'netProfit', 'averageNetProfit', 'NAR',
                     'profitFactor', 'recoveryFactor', 'MDDLength',
                     'selfMDD', 'winRate', 'tradingTimeRatio',
                     'averageTradingBar', 'tradeNumber', 'maxValue',
                     'minValue', 'initialv', 'totalCommission',
                     'barNumber', 'expectancy100', 'bnhReturn',
                     'bnhRatio']
    simulations.columns = reportColumns
    consoResult = cr.scoring(simulations)
    consoResult = consoResult.sort_values(['res'], ascending=False)
    consoResult.sort_values('res').to_csv('{}/optReport.csv'.format(params['resultLocation']), index=False)

    timeRequired = time.time() - startTime
    print('timeRequired={:.2f}s'.format(timeRequired))

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2016, 10, 31)
# Different stocks from past posts because of different data source (no plot for NTDOY)
symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "YHOO", "SNY", "VZ", "IBM", "HPQ", "QCOM", "NVDA"]
datafeeds = {s: web.DataReader(s, "google", start, end) for s in symbols}
for df in datafeeds.values():
    df["OpenInterest"] = 0  # PandasData reader expects an OpenInterest column;
# not provided by Google and we don't use it so set to 0

cerebro = bt.Cerebro(stdstats=False)

plot_symbols = ["AAPL", "GOOG", "NVDA"]
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
        data.plotinfo.plot = False
    cerebro.adddata(data)  # Give the data to cerebro

cerebro.broker.setcash(1000000)
cerebro.broker.setcommission(0.02)
cerebro.addstrategy(SMAC)
cerebro.addobserver(AcctValue)
cerebro.addobservermulti(bt.observers.BuySell)  # Plots up/down arrows
cerebro.addsizer(PropSizer)
cerebro.addanalyzer(AcctStats)

cerebro.run()
