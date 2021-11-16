#!/usr/bin/env python3

import time
import backtrader as bt
import datetime as dt
import yaml
from ccxtbt import CCXTStore
from config import BINANCE, ENV, PRODUCTION, COIN_TARGET, COIN_REFER, DEBUG

from dataset.dataset import CustomDataset
from sizer.percent import FullMoney
from strategies.basic_rsi import BasicRSI
from utils import print_trade_analysis, print_sqn, send_telegram_message

from PrepareCSV import prepare_data
import toolkit as tk

def main():
    cerebro = bt.Cerebro(quicknotify=True)

    if ENV == PRODUCTION:  # Live trading with Binance
        broker_config = {
            'apiKey': BINANCE.get("key"),
            'secret': BINANCE.get("secret"),
            'timeout': 5000,
            'verbose': False,
            'nonce': lambda: str(int(time.time() * 1000)),
            'enableRateLimit': True,
        }

        store = CCXTStore(exchange='binanceusdm', currency=COIN_REFER, config=broker_config, retries=5, debug=False,
                          sandbox=True)

        broker_mapping = {
            'order_types': {
                bt.Order.Market: 'market',
                bt.Order.Limit: 'limit',
                bt.Order.Stop: 'stop-loss',
                bt.Order.StopLimit: 'stop limit'
            },
            'mappings': {
                'closed_order': {
                    'key': 'status',
                    'value': 'closed'
                },
                'canceled_order': {
                    'key': 'status',
                    'value': 'canceled'
                }
            }
        }
        broker = store.getbroker(broker_mapping=broker_mapping)
        cerebro.setbroker(broker)

        hist_start_date = dt.datetime.utcnow() - dt.timedelta(minutes=3000)
        with open('dataset/symbol_config.yaml', mode='r') as f:
            symbol_config = f.read()
            symbol_config = yaml.load(symbol_config, Loader=yaml.FullLoader)
            for symbol in symbol_config.keys():
                datakl = store.getdata(
                    dataname=f'{symbol}',
                    name=f'{symbol}',
                    timeframe=bt.TimeFrame.Minutes,
                    fromdate=hist_start_date,
                    compression=1,
                    ohlcv_limit=10000
                )
                dataha = datakl.clone()
                dataha.addfilter(bt.filters.HeikinAshi(dataha))
                # Add the feed
                cerebro.adddata(datakl, name=f'{symbol}_Kline')
                cerebro.adddata(dataha, name=f'{symbol}_Heikin')

    else:  # Backtesting with CSV file
        with open('dataset/symbol_config.yaml', mode='r') as f:
            symbol_config = f.read()
            symbol_config = yaml.load(symbol_config, Loader=yaml.FullLoader)
            for symbol in symbol_config.keys():
                t0str, t9str = '2021-09-01', '2021-10-01'
                fgCov = False
                df = prepare_data(t0str, t9str, symbol, fgCov=fgCov, prep_new=True, mode='test')
                datakl = tk.pools_get4df(df, t0str, t9str, fgCov=fgCov)
                dataha = datakl.clone()
                dataha.addfilter(bt.filters.HeikinAshi(dataha))
                cerebro.resampledata(datakl, name=f'{symbol}_10m_Kline', timeframe=bt.TimeFrame.Minutes, compression=10)
                cerebro.resampledata(dataha, name=f'{symbol}_10m_Heikin', timeframe=bt.TimeFrame.Minutes, compression=10)
            f.close()
        broker = cerebro.getbroker()
        broker.setcommission(commission=0.0004, name=COIN_TARGET)  # Simulating exchange fee
        broker.setcash(100000.0)
        cerebro.addsizer(FullMoney)

    # Analyzers to evaluate trades and strategies
    # SQN = Average( profit / risk ) / StdDev( profit / risk ) x SquareRoot( number of trades )
    # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    # cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

    # Include Strategy
    strategy = []
    strategy_list = []
    for symbol in symbol_config.keys():
        strategy_list = list(set(strategy).union(set(symbol.keys())))
    for strategy in strategy_list:
        cerebro.addstrategy(strategy)
    # cerebro.broker.set_checksubmit(False)
    # Starting backtrader bot
    initial_value = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % initial_value)
    result = cerebro.run()
    # Print analyzers - results
    final_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_value)
    print('Profit %.3f%%' % ((final_value - initial_value) / initial_value * 100))
    print_trade_analysis(result[0].analyzers.ta.get_analysis())
    print_sqn(result[0].analyzers.sqn.get_analysis())
    if DEBUG:
        cerebro.plot()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("finished.")
        time = dt.datetime.now().strftime("%d-%m-%y %H:%M")
        send_telegram_message("Bot finished by user at %s" % time)
    except Exception as err:
        send_telegram_message("Bot finished with error: %s" % err)
        print("Finished with error: ", err)
        raise
