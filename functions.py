import pandas as pd
from datetime import datetime
from decimal import Decimal
import ccxt
import yaml
import json
import time
import h5py
import sys

pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('mode.chained_assignment', None)

# 从配置文件中加载各种配置
yaml_path = 'data//settings.yaml'
with open(yaml_path, 'r') as f:
    st = f.read()
    data = yaml.load(st, Loader=yaml.FullLoader)
    f.close()
# 配置信息
default_reduce_rate = data['default_reduce_rate']
default_leverage_rate = data['default_leverage_rate']
pin = data['pin']
test_info = data['test_info']
test_record = data['test_record']
binance_order_types = data['binance_order_types']
BINANCE_CONFIG = data['BINANCE_CONFIG']
from_number = data['From_Number']
to_number = data['To_Number']
twilio_key = data['twilio_key']
twilio_tkn = data['twilio_token']
Max_atp = int(data['maximum_number_of_attempts'])
exchange = ccxt.binance(BINANCE_CONFIG)
# exchange.set_sandbox_mode(True)
# 变量
strategy_list = ['']
strategy_symbol_list = ['']
strategy_symbol_time_period_list = ['']
reduce_rate_list = ['']
leverage_rate_list = ['']


def load_config(strategy, symbol, time_period):
    # 从配置文件中加载各种配置
    with open(yaml_path, 'r') as f:
        global data
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
        f.close()
    if {'strategy_list', f'{strategy}_symbol_list', f'{strategy}_{symbol}_time_period_list', f'{strategy}_{symbol}_{time_period}_reduce_rate', f'{strategy}_{symbol}_{time_period}_leverage_rate'}.issubset(data.keys()):
        global strategy_list
        strategy_list = data['strategy_list']
        global strategy_symbol_list
        strategy_symbol_list = [x for x in data.keys() if 'symbol_list' in x]
        global strategy_symbol_time_period_list
        strategy_symbol_time_period_list = [x for x in data.keys() if 'time_period_list' in x]
        global reduce_rate_list
        reduce_rate_list = [x for x in data.keys() if 'reduce_rate' in x].remove('default_reduce_rate')
        global leverage_rate_list
        leverage_rate_list = [x for x in data.keys() if 'leverage_rate' in x].remove('default_leverage_rate')
    else:
        _list = ['strategy_list', f'{strategy}_symbol_list', f'{strategy}_{symbol}_time_period_list', f'{strategy}_{symbol}_{time_period}_reduce_rate', f'{strategy}_{symbol}_{time_period}_leverage_rate']
        with open(yaml_path, 'w') as f:
            for x in _list:
                if x not in data.keys():
                    data[x] = ['']
            yaml.dump(data, f)
            f.close()


def check_signal(strategy, symbol, time_period, signal_type):
    """
    功能时用于检查每次收到的信号是否在预设文件中，如果没有，则在预设文件中新增，并且在数据库文件中新增对应位置来初始化
    """
    global data
    info = test_info
    load_config(strategy, symbol, time_period)
    with open(yaml_path, 'r') as f:
        if ('strategy_list' not in data.keys()) or (strategy not in data['strategy_list']):
            if 'reduce' in signal_type:
                if data['strategy_list'] == ['']:
                    del data['strategy_list']
                result = 'rejected'
            if 'close' in signal_type:
                if data['strategy_list'] == ['']:
                    del data['strategy_list']
                result = 'rejected'
            else:
                data['strategy_list'].append(strategy)
                data['strategy_list'] = list(set(data['strategy_list']))
                if '' in data['strategy_list']:
                    data['strategy_list'].remove('')
                h = h5py.File(f'data//{strategy}.h5', mode='w')
                h_record = h5py.File(f'data//{strategy}_trading_record.h5', mode='w')
                h.close()
                h_record.close()
                result = 'passed'
        if (f'{strategy}_symbol_list' not in data.keys()) or (symbol not in data[f'{strategy}_symbol_list']):
            if 'reduce' in signal_type:
                result = 'rejected'
                if data[f'{strategy}_symbol_list'] == ['']:
                    del data[f'{strategy}_symbol_list']
            if 'close' in signal_type:
                result = 'rejected'
                if data[f'{strategy}_symbol_list'] == ['']:
                    del data[f'{strategy}_symbol_list']
            else:
                temp_record = pd.DataFrame(data['test_record']).astype(str)
                temp_record.to_hdf(f'data//{strategy}_trading_record.h5', key=f'{symbol}', mode='a', format='t')
                data[f'{strategy}_symbol_list'].append(symbol)
                data[f'{strategy}_symbol_list'] = list(set(data[f'{strategy}_symbol_list']))
                if '' in data[f'{strategy}_symbol_list']:
                    data[f'{strategy}_symbol_list'].remove('')
                temp_info = info.copy()
                temp_info['time_period'] = 'none'
                temp_info = pd.DataFrame(temp_info).astype(str)
                temp_info.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a', format='t')
                result = 'passed'
            # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
            # print(df_02, strategy, symbol, time_period)
            # print('行号: ', str(sys._getframe().f_lineno))
        if (f'{strategy}_{symbol}_time_period_list' not in data.keys()) or (time_period not in data[f'{strategy}_{symbol}_time_period_list']):
            if 'reduce' in signal_type:
                result = 'rejected'
                if data[f'{strategy}_{symbol}_time_period_list'] == ['']:
                    del data[f'{strategy}_{symbol}_time_period_list']
                if data[f'{strategy}_{symbol}_{time_period}_reduce_rate'] == ['']:
                    del data[f'{strategy}_{symbol}_{time_period}_reduce_rate']
                if data[f'{strategy}_{symbol}_{time_period}_leverage_rate'] == ['']:
                    del data[f'{strategy}_{symbol}_{time_period}_leverage_rate']
            if 'close' in signal_type:
                result = 'rejected'
                if data[f'{strategy}_{symbol}_time_period_list'] == ['']:
                    del data[f'{strategy}_{symbol}_time_period_list']
                if data[f'{strategy}_{symbol}_{time_period}_reduce_rate'] == ['']:
                    del data[f'{strategy}_{symbol}_{time_period}_reduce_rate']
                if data[f'{strategy}_{symbol}_{time_period}_leverage_rate'] == ['']:
                    del data[f'{strategy}_{symbol}_{time_period}_leverage_rate']
            else:
                data[f'{strategy}_{symbol}_time_period_list'].append(time_period)
                data[f'{strategy}_{symbol}_time_period_list'] = list(set(data[f'{strategy}_{symbol}_time_period_list']))
                if '' in data[f'{strategy}_{symbol}_time_period_list']:
                    data[f'{strategy}_{symbol}_time_period_list'].remove('')
                data[f'{strategy}_{symbol}_{time_period}_reduce_rate'] = default_reduce_rate
                data[f'{strategy}_{symbol}_{time_period}_leverage_rate'] = default_leverage_rate
                if '' in data[f'{strategy}_{symbol}_{time_period}_reduce_rate']:
                    data[f'{strategy}_{symbol}_{time_period}_reduce_rate'].remove('')
                if '' in data[f'{strategy}_{symbol}_{time_period}_leverage_rate']:
                    data[f'{strategy}_{symbol}_{time_period}_leverage_rate'].remove('')
                df = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
                df = pd.DataFrame(df)
                if 'time_period' in df.columns:
                    df.set_index(['time_period'], inplace=True)
                info['time_period'] = time_period
                info['symbol'] = symbol
                info = pd.DataFrame(info)
                info.set_index(['time_period'], inplace=True)
                info = pd.DataFrame(info).astype(str)
                df = df.append(info)
                if 'none' in df.index:
                    df = df.drop(['none'])
                # df = df[~df.index.duplicated(keep='first')]
                df = df.astype(str)
                df.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
                # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
                # print(df_02, strategy, symbol, time_period)
                # print('行号: ', str(sys._getframe().f_lineno))
                result = 'passed'
        if f'{strategy}_{symbol}_{time_period}_leverage_rate' not in data.keys():
            if 'reduce' in signal_type:
                result = 'rejected'
                if data[f'{strategy}_{symbol}_{time_period}_leverage_rate'] == ['']:
                    del data[f'{strategy}_{symbol}_{time_period}_leverage_rate']
            if 'close' in signal_type:
                result = 'rejected'
                if data[f'{strategy}_{symbol}_{time_period}_leverage_rate'] == ['']:
                    del data[f'{strategy}_{symbol}_{time_period}_leverage_rate']

            else:
                data[f'{strategy}_{symbol}_{time_period}_leverage_rate'] = default_leverage_rate
                if '' in data[f'{strategy}_{symbol}_{time_period}_leverage_rate']:
                    data[f'{strategy}_{symbol}_{time_period}_leverage_rate'].remove('')
                    result = 'passed'
        else:
            result = 'passed'
        f.close()
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
        f.close()
    return result


def schedule_sync():
    """
    定时同步离线资金
    """
    latest_balance = get_latest_balance()
    sync(latest_balance)
    cal_allocated_ratio()


def position_sync():
    """
    更新仓位
    """
    """
    思路：
    遍历所有策略，以及策略之下的交易对
    生成f'{symbol}_long, {symbol}_short'
    遍历所有记录文件，遍历所有记录文件下的symbol，遍历所有strategy_symbol_time_period, 将对应金额相加给{strategy}_{symbol}_long, {strategy}_{symbol}_short
    """
    D = {}
    exchange_info = exchange.fapiPublicGetExchangeinfo()
    df_ex = pd.DataFrame(exchange_info['symbols'])
    df_ex = df_ex[['symbol', 'pricePrecision', 'quantityPrecision']]
    df_ex.set_index('symbol', inplace=True)
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            D[f'{s}_LONG'] = Decimal('0.000')
            D[f'{s}_SHORT'] = Decimal('0.000')
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            df = pd.DataFrame(df).astype(str)
            if 'time_period' in df.columns:
                df.set_index(['time_period'], inplace=True)
            for t in data[f'{S}_{s}_time_period_list']:
                D[f'{s}_LONG'] += Decimal(df.loc[t, 'period_LONG_position'])
                D[f'{s}_SHORT'] += Decimal(df.loc[t, 'period_SHORT_position'])
    symbol_list = []
    for S in data['strategy_list']:
        symbol_list = list(set(symbol_list).union(set(data[f'{S}_symbol_list'])))
    for s in symbol_list:
        price_precision = df_ex.at[s, 'pricePrecision']
        quantity_precision = df_ex.at[s, 'quantityPrecision']
        quantity_precision = '%.0{}f'.format(quantity_precision) % 0
        price_precision = '%.0{}f'.format(price_precision) % 0
        latest_price = get_ticker_price(s)
        res = exchange.fapiPrivateGetAccount()
        res = pd.DataFrame(res['positions'])
        df_SHORT = res[res['positionSide'].isin(['SHORT'])]
        df_SHORT = df_SHORT[df_SHORT['symbol'].isin([f'{s}'])]
        df_SHORT = df_SHORT[df_SHORT.symbol.apply(lambda x: x.isalpha())]
        df_SHORT.set_index('symbol', inplace=True)
        df_SHORT = df_SHORT.loc[s, 'positionAmt']
        df_LONG = res[res['positionSide'].isin(['LONG'])]
        df_LONG = df_LONG[df_LONG['symbol'].isin([f'{s}'])]
        df_LONG = df_LONG[df_LONG.symbol.apply(lambda x: x.isalpha())]
        df_LONG.set_index('symbol', inplace=True)
        df_LONG = df_LONG.loc[s, 'positionAmt']
        adjust_LONG = abs(Decimal(D[f'{s}_LONG'])) - abs(Decimal(df_LONG))
        # print(f'{s} {adjust_LONG}')
        adjust_LONG = modify_order_quantity(quantity_precision, adjust_LONG)
        if (adjust_LONG > Decimal(0)) and ((latest_price * adjust_LONG) > 10):
            # print(f'{s} lc LONG', D[f'{s}_LONG'])
            # print(f'{s} ex LONG {df_LONG}')
            # print(f'{s} adjust LONG {adjust_LONG}')
            signal_type = 'open_LONG'
            adjust_LONG = abs(adjust_LONG)
            order = post_order(s, signal_type, adjust_LONG)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{s}: {signal_type} Position {executedQty} at {avgPrice}'
            # msg = rec01 + rec02
            # send_message(msg)
        if adjust_LONG < Decimal(0):
            # print(f'{s} lc LONG', D[f'{s}_LONG'])
            # print(f'{s} ex LONG {df_LONG}')
            # print(f'{s} adjust LONG {adjust_LONG}')
            signal_type = 'reduce_LONG'
            adjust_LONG = abs(adjust_LONG)
            order = post_order(s, signal_type, adjust_LONG)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{s}: {signal_type} Position {executedQty} at {avgPrice}'
            # msg = rec01 + rec02
            # send_message(msg)
        adjust_SHORT = abs(Decimal(D[f'{s}_SHORT'])) - abs(Decimal(df_SHORT))
        adjust_SHORT = modify_order_quantity(quantity_precision, adjust_SHORT)
        if (adjust_SHORT > Decimal(0)) and ((latest_price * adjust_SHORT) > 10):
            # print(f'{s} lc SHORT', D[f'{s}_SHORT'])
            # print(f'{s} ex SHORT {df_SHORT}')
            # print(f'{s} adjust SHORT {adjust_SHORT}')
            signal_type = 'open_SHORT'
            adjust_SHORT = abs(adjust_SHORT)
            order = post_order(s, signal_type, adjust_SHORT)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{s}: {signal_type} Position {executedQty} at {avgPrice}'
            # msg = rec01 + rec02
            # send_message(msg)
        if adjust_SHORT < Decimal(0):
            # print(f'{s} lc SHORT', D[f'{s}_SHORT'])
            # print(f'{s} ex SHORT {df_SHORT}')
            # print(f'{s} adjust SHORT {adjust_SHORT}')
            signal_type = 'reduce_SHORT'
            adjust_SHORT = abs(adjust_SHORT)
            order = post_order(s, signal_type, adjust_SHORT)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{s}: {signal_type} Position {executedQty} at {avgPrice}'
            # msg = rec01 + rec02
            # send_message(msg)
        time.sleep(1)

def usdt_future_exchange_info(symbol):
    """
    获取交易币种的最小下单价格、数量精度
    """
    n = 0
    while True:
        try:
            exchange_info = exchange.fapiPublicGetExchangeinfo()
            break
        except Exception:
            time.sleep(1)
            if n >= Max_atp:
                print('If you encounter difficulties, just don\'t do it and get a good night\'s sleep'.center(120))
                msg = 'Emergent Issue Occurred Boss, Please Check the Server!'
                send_message(msg)
                break
            else:
                n += 1
                continue
    # 转化为dataframe
    df = pd.DataFrame(exchange_info['symbols'])
    df = df[['symbol', 'pricePrecision', 'quantityPrecision']]
    df.set_index('symbol', inplace=True)
    price_precision = df.at[symbol, 'pricePrecision']
    quantity_precision = df.at[symbol, 'quantityPrecision']
    # symbol_temp[symbol]['最小下单量精度'] = None if p == 0 else int(p)
    return price_precision, quantity_precision


def get_ticker_price(symbol):
    n = 0
    while True:
        try:
            latest_price = exchange.fapiPublic_get_ticker_price({"symbol": f"{symbol}"})['price']
            break
        except Exception:
            time.sleep(1)
            if n >= Max_atp:
                print('If you encounter difficulties, just don\'t do it and get a good night\'s sleep'.center(120))
                msg = 'Emergent Issue Occurred Boss, Please Check the Server!'
                send_message(msg)
                break
            else:
                n += 1
                continue

    latest_price = Decimal(latest_price)
    return latest_price


def modify_order_quantity(quantity_precision, quantity):
    """
    根据交易所的精度限制（最小下单单位、量等），修改下单的数量和价格
    """
    # 根据每个币种的精度，修改下单数量的精度
    # 获取每个币对的精度
    quantity = Decimal(quantity).quantize(Decimal(quantity_precision))
    return quantity


def get_precision(symbol):
    """
    对接交易所，获取币对的数量和价格精度
    输出 0.000 这样的小数字符串形式
    """
    price_precision, quantity_precision = usdt_future_exchange_info(symbol)
    quantity_precision = '%.0{}f'.format(quantity_precision) % 0
    price_precision = '%.0{}f'.format(price_precision) % 0
    return price_precision, quantity_precision


def modify_decimal(n):
    """
    快速调用的将小数做Decimal处理的小功能
    """
    n = float(n)
    n = Decimal(n).quantize(Decimal("0.000000"))
    return n


def get_latest_balance():
    # with open("data//response.json", mode='r') as response:
    #     response = json.load(response)
    #     account_info = response
    n = 0
    while True:
        try:
            response = exchange.fapiPrivateGetAccount()
            break
        except Exception:
            time.sleep(1)
            if n >= Max_atp:
                print('If you encounter difficulties, just don\'t do it and get a good night\'s sleep'.center(120))
                msg = 'Emergent Issue Occurred Boss, Please Check the Server!'
                send_message(msg)
                break
            else:
                n += 1
                continue

    account_info = response

    # 获取账户当前总资金
    assets_df = pd.DataFrame(account_info['assets'])
    assets_df = assets_df.set_index('asset')
    latest_balance = Decimal(assets_df.loc['USDT', 'marginBalance'])  # 保证金余额
    return latest_balance


def join(strategy, symbol, time_period):
    """
    当有新交易策略/交易对/交易时间区间出现时使用, 利用原有allocate_ratio来对新加入的部分进行分配
    """
    # 初始化当中, 是以allocate ratio 决定allocate funds
    # 初始化
    L = 0
    df = []
    load_config(strategy, symbol, time_period)
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='r')
            df = pd.DataFrame(df).astype(str)
            if 'time_period' in df.columns:
                df.set_index(['time_period'], inplace=True)
            L += len(df.index)
    L = Decimal(L)
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            df = pd.DataFrame(df).astype(str)
            if 'time_period' in df.columns:
                df.set_index(['time_period'], inplace=True)
            for t in data[f'{S}_{s}_time_period_list']:
                if {S, s, t}.issubset([strategy, symbol, time_period]):
                    pass
                else:
                    n = df.loc[t, 'period_allocated_ratio']
                    n = modify_decimal(n)
                    n *= modify_decimal((L - Decimal(1)) / L)
                    df.loc[t, 'period_allocated_ratio'] = n
            df = df.astype(str)
            df.to_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
    df = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
    df = pd.DataFrame(df).astype(str)
    if 'time_period' in df.columns:
        df.set_index(['time_period'], inplace=True)
    df.loc[time_period, 'schedule_action'] = 'none'
    df.loc[time_period, 'period_allocated_ratio'] = modify_decimal(1 / L)
    df = df.astype(str)
    df.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
    # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
    # print(df_02, strategy, symbol, time_period)
    # print('行号: ', str(sys._getframe().f_lineno))
    # 编辑好各symbol各period_allocated


def sync(latest_balance):
    """
    属于定时任务, 定期更新最新资金, 使用allocated_ratio来进行分配
    """
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            df = pd.DataFrame(df).astype(str)
            if 'time_period' in df.columns:
                df.set_index(['time_period'], inplace=True)
            for t in data[f'{S}_{s}_time_period_list']:
                n = df.loc[t, 'period_allocated_ratio']
                period_allocated_ratio = modify_decimal(n)
                df.loc[t, 'period_allocated_funds'] = modify_decimal(latest_balance * period_allocated_ratio)
            df = df.astype(str)
            df.to_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            # df_02 = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='r')
            # print(df_02, S, s, t)
            # print('行号: ', str(sys._getframe().f_lineno))


def cal_allocated_ratio():
    """
    用于通常情况下的资金分配, 用当前策略的allocated_funds来计算出allocated_ratio
    """
    # 累加出symbol_allocated_funds和strategy_allocated_funds
    account_balance = Decimal('0.000')
    for S in data['strategy_list']:
        strategy_allocated_funds = Decimal('0.000')
        for s in data[f'{S}_symbol_list']:
            symbol_allocated_funds = Decimal('0.000')
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='r')
            df = pd.DataFrame(df).astype(str)
            for t in data[f'{S}_{s}_time_period_list']:
                funds = df.loc[t, 'period_allocated_funds']
                funds = modify_decimal(funds)
                symbol_allocated_funds += funds
            strategy_allocated_funds += symbol_allocated_funds
        account_balance += strategy_allocated_funds
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            df = pd.DataFrame(df).astype(str)
            if 'time_period' in df.columns:
                df.set_index(['time_period'], inplace=True)
            # 通过allocated_funds来逐个决定_allocated_ratio
            for t in data[f'{S}_{s}_time_period_list']:
                df.loc[t, 'account_balance'] = account_balance
                period_allocated_funds = df.loc[t, 'period_allocated_funds']
                period_allocated_funds = modify_decimal(period_allocated_funds)
                df.loc[t, 'period_allocated_ratio'] = modify_decimal(period_allocated_funds / account_balance)
            df = df.astype(str)
            df.to_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            # df_02 = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='r')
            # print(df_02, S, s, t)
            # print('行号: ', str(sys._getframe().f_lineno))


def remove(strategy, symbol, time_period):
    """
    用于当需要移除交易对的情况, 在配置文件以及数据库文件中都删除其信息
    """
    with open(yaml_path, "w") as f:
        yf = yaml.load(f)
        if 'remove' in strategy:
            strategy = strategy.replace('remove_', '')
            del yf['strategy_list'][f'{strategy}']
            yaml.dump(yf, f)
            yf.close()
            with h5py.File(f'data//{strategy}.h5', "w") as f:
                f.close()
        if 'remove' in symbol:
            symbol = symbol.replace('remove_', '')
            del yf[f'{strategy}_symbol_list'][f'{symbol}']
            yaml.dump(yf, f)
            yf.close()
            with h5py.File(f'data//{strategy}.h5', "a") as f:
                del f[f'{symbol}']
                f.close()
        if 'remove' in time_period:
            symbol = time_period.replace('remove_', '')
            del yf[f'{strategy}_{symbol}_time_period_list'][f'{symbol}']
            yaml.dump(yf, f)
            yf.close()
            df = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
            df = pd.DataFrame(df).astype(str)
            del df[f'{time_period}']
            df = df.astype(str)
            df.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
            # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
            # print(df_02, strategy, symbol, time_period)
            # print('行号: ', str(sys._getframe().f_lineno))
    with open(yaml_path, 'r') as f:
        global data
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
        f.close()
    L = 0
    par = Decimal('0.000')
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='r')
            df = pd.DataFrame(df).astype(str)
            if 'time_period' in df.columns:
                df.set_index(['time_period'], inplace=True)
            for t in data[f'{S}_{s}_time_period_list']:
                p = df.loc[t, 'period_allocated_ratio']
                p = modify_decimal(p)
                par += p
    for S in data['strategy_list']:
        for s in data[f'{S}_symbol_list']:
            df = pd.read_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            df = pd.DataFrame(df).astype(str)
            if 'time_period' in df.columns:
                df.set_index(['time_period'], inplace=True)
            for t in data[f'{S}_{s}_time_period_list']:
                n = df.loc[t, 'period_allocated_ratio']
                n = modify_decimal(n)
                n *= modify_decimal(1 / par)
                df.loc[t, 'period_allocated_ratio'] = n
            df = df.astype(str)
            df.to_hdf(f'data//{S}.h5', key=f'{s}', mode='a')
            # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
            # print(df_02, strategy, symbol, time_period)
            # print('行号: ', str(sys._getframe().f_lineno))
    latest_balance = get_latest_balance()
    sync(latest_balance)
    cal_allocated_ratio()


def update_allocation_statistics(strategy, symbol, time_period):
    """
    通过初始化类别，更新资金分配状况，和计算分配比例
    """
    # ====调用接口====
    # exchange.fapiPrivateGetAccount()
    # with open("data//response.json", mode='r') as response:
    #     response = json.load(response)
    #     account_info = response
    latest_balance = get_latest_balance()
    # ====更新离线数据====
    df = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
    df = pd.DataFrame(df).astype(str)
    if 'time_period' in df.columns:
        df.set_index(['time_period'], inplace=True)
    if df.loc[time_period, 'schedule_action'] == 'join':
        join(strategy, symbol, time_period)
        sync(latest_balance)
        cal_allocated_ratio()
    elif df.loc[time_period, 'schedule_action'] == 'sync':
        sync(latest_balance)
        cal_allocated_ratio()
    elif df.loc[time_period, 'schedule_action'] == 'none':
        cal_allocated_ratio()


def position_management(signal_type, strategy, symbol, time_period, quantity, trading_info, quantity_precision):
    """
    根据订单类型来来得出开仓量, 和更新数据库文件中的对应记录持仓量
    """
    quantity = Decimal(quantity)
    if signal_type == 'reduce_SHORT':
        # 计算出减仓量
        reduce_rate = data[f'{strategy}_{symbol}_{time_period}_reduce_rate']
        reduce_rate = Decimal(reduce_rate[0])
        # print(reduce_rate, type(reduce_rate))
        # print(quantity, type(quantity))
        reduce_quantity = Decimal(reduce_rate * quantity)
        reduced_quantity = Decimal(quantity - reduce_quantity)
        # 在数据库文件里编辑
        trading_info.loc[time_period, 'period_SHORT_position'] = modify_order_quantity(quantity_precision, reduced_quantity)
        quantity = reduce_quantity
    if signal_type == 'reduce_LONG':
        # 计算出减仓量
        reduce_rate = data[f'{strategy}_{symbol}_{time_period}_reduce_rate']
        reduce_rate = Decimal(reduce_rate[0])
        reduce_quantity = Decimal(reduce_rate * quantity)
        reduced_quantity = Decimal(quantity - reduce_quantity)
        # 在数据库文件里编辑
        trading_info.loc[time_period, 'period_LONG_position'] = modify_order_quantity(quantity_precision, reduced_quantity)
        quantity = reduce_quantity
    if signal_type == 'open_LONG':
        orig_quantity = trading_info.loc[time_period, 'period_LONG_position']
        orig_quantity = Decimal(orig_quantity)
        lev_rate = data[f'{strategy}_{symbol}_{time_period}_leverage_rate']
        lev_rate = Decimal(lev_rate[0])
        quantity *= Decimal(lev_rate)
        trading_info.loc[time_period, 'period_LONG_position'] = modify_order_quantity(quantity_precision, orig_quantity + quantity)
    if signal_type == 'open_SHORT':
        orig_quantity = trading_info.loc[time_period, 'period_SHORT_position']
        orig_quantity = Decimal(orig_quantity)
        lev_rate = data[f'{strategy}_{symbol}_{time_period}_leverage_rate']
        lev_rate = Decimal(lev_rate[0])
        quantity *= Decimal(lev_rate)
        trading_info.loc[time_period, 'period_SHORT_position'] = modify_order_quantity(quantity_precision, orig_quantity + quantity)
    if signal_type == 'close_LONG':
        quantity = trading_info.loc[time_period, 'period_LONG_position']
        quantity = Decimal(quantity)
        trading_info.loc[time_period, 'period_LONG_position'] = modify_order_quantity(quantity_precision, 0)
    if signal_type == 'close_SHORT':
        quantity = trading_info.loc[time_period, 'period_SHORT_position']
        quantity = Decimal(quantity)
        trading_info.loc[time_period, 'period_SHORT_position'] = modify_order_quantity(quantity_precision, 0)
    trading_info = trading_info.astype(str)
    trading_info.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
    return quantity


def processing_trading_action(strategy, symbol, time_period, signal_type):
    """
    处理交易信号，计算开仓量，发送订单
    """
    trading_info = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
    trading_info = pd.DataFrame(trading_info).astype(str)
    if 'time_period' in trading_info.columns:
        trading_info.set_index(['time_period'], inplace=True)
    price_precision, quantity_precision = get_precision(symbol)
    latest_price = get_ticker_price(symbol)
    if signal_type in ['reduce_LONG', 'close_LONG']:
        reduce_quantity = trading_info.loc[time_period, 'period_LONG_position']
        # reduce_quantity = modify_order_quantity(quantity_precision, reduce_quantity)
        reduce_quantity = position_management(signal_type, strategy, symbol, time_period, reduce_quantity, trading_info, quantity_precision)
        reduce_quantity = modify_order_quantity(quantity_precision, reduce_quantity)
        if reduce_quantity > 0:
            if reduce_quantity > Decimal(0):
                order = post_order(symbol, signal_type, reduce_quantity)
                trading_record(order, strategy, symbol, time_period, signal_type)
                processing_record(strategy, symbol, time_period, signal_type, order)
                status = order['status']
                orderId = order['orderId']
                avgPrice = order['avgPrice']
                executedQty = order['executedQty']
                rec01 = f'{status} Order : # {orderId} #'
                rec02 = f'{symbol}: {signal_type} Position {executedQty} at {avgPrice}'
                msg = rec01 + rec02
                send_message(msg)
            else:
                print('Order quantity is less than $10 or below the precision!'.center(120))
                print('Future Position Did Not Adjust Properly!'.center(120))
                a = 'Order quantity is less than $10 or below the precision!'
                b = 'Future Position Did Not Adjust Properly!'
                send_message(f'{a} {b}')
        else:
            print(f'Rejected Signal You didn\'t Open Position Named {symbol} on {time_period}'.center(120))
            a = f'Rejected Signal You didn\'t Open Position Named {symbol} on {time_period}'
            # send_message(a)

    if signal_type in ['reduce_SHORT', 'close_SHORT']:
        reduce_quantity = trading_info.loc[time_period, 'period_SHORT_position']
        # reduce_quantity = modify_order_quantity(price_precision, reduce_quantity)
        reduce_quantity = position_management(signal_type, strategy, symbol, time_period, reduce_quantity, trading_info, quantity_precision)
        reduce_quantity = modify_order_quantity(quantity_precision, reduce_quantity)
        if reduce_quantity > 0:
            if reduce_quantity > Decimal(0):
                order = post_order(symbol, signal_type, reduce_quantity)
                trading_record(order, strategy, symbol, time_period, signal_type)
                processing_record(strategy, symbol, time_period, signal_type, order)
                status = order['status']
                orderId = order['orderId']
                avgPrice = order['avgPrice']
                executedQty = order['executedQty']
                rec01 = f'{status} Order : # {orderId} #'
                rec02 = f'{symbol}: {signal_type} Position {executedQty} at {avgPrice}'
                msg = rec01 + rec02
                send_message(msg)
            else:
                print('Order quantity is less than $10 or below the precision!'.center(120))
                print('Future Position Did Not Adjust Properly!'.center(120))
                a = 'Order quantity is less than $10 or below the precision!'
                b = 'Future Position Did Not Adjust Properly!'
                send_message(f'{a} {b}')
        else:
            print(f'Rejected Signal You didn\'t Open Position Named {symbol} on {time_period}'.center(120))
            a = f'Rejected Signal You didn\'t Open Position Named {symbol} on {time_period}'
            # send_message(a)
    if signal_type == 'open_LONG':
        reduce_quantity = trading_info.loc[time_period, 'period_SHORT_position']
        reduce_quantity = position_management('close_SHORT', strategy, symbol, time_period, reduce_quantity, trading_info, quantity_precision)
        reduce_quantity = modify_order_quantity(quantity_precision, reduce_quantity)
        if reduce_quantity > Decimal(0):
            order = post_order(symbol, 'close_SHORT', reduce_quantity)
            trading_record(order, strategy, symbol, time_period, 'close_SHORT')
            processing_record(strategy, symbol, time_period, 'close_SHORT', order)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{symbol}: {signal_type} Position {executedQty} at {avgPrice}'
            msg = rec01 + rec02
            send_message(msg)
        n = trading_info.loc[time_period, 'period_allocated_funds']
        allocated_funds = modify_decimal(n)
        quantity = allocated_funds / latest_price
        quantity = position_management(signal_type, strategy, symbol, time_period, quantity, trading_info, quantity_precision)
        quantity = modify_order_quantity(quantity_precision, quantity)
        if (quantity > Decimal(0)) and ((latest_price * quantity) > 10):
            order = post_order(symbol, signal_type, quantity)
            trading_record(order, strategy, symbol, time_period, signal_type)
            processing_record(strategy, symbol, time_period, signal_type, order)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{symbol}: {signal_type} Position {executedQty} at {avgPrice}'
            msg = rec01 + rec02
            send_message(msg)
        else:
            print('Order quantity is less than $10 or below the precision!'.center(120))
            print('Future Position Did Not Adjust Properly!'.center(120))
            a = 'Order quantity is less than $10 or below the precision!'
            b = 'Future Position Did Not Adjust Properly!'
            send_message(f'{a} {b}')
    if signal_type == 'open_SHORT':
        reduce_quantity = trading_info.loc[time_period, 'period_LONG_position']
        reduce_quantity = position_management('close_LONG', strategy, symbol, time_period, reduce_quantity, trading_info, quantity_precision)
        reduce_quantity = modify_order_quantity(quantity_precision, reduce_quantity)
        if reduce_quantity > Decimal(0):
            order = post_order(symbol, 'close_LONG', reduce_quantity)
            trading_record(order, strategy, symbol, time_period, 'close_LONG')
            processing_record(strategy, symbol, time_period, 'close_LONG', order)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{symbol}: {signal_type} Position {executedQty} at {avgPrice}'
            msg = rec01 + rec02
            send_message(msg)
        n = trading_info.loc[time_period, 'period_allocated_funds']
        allocated_funds = modify_decimal(n)
        quantity = allocated_funds / latest_price
        quantity = position_management(signal_type, strategy, symbol, time_period, quantity, trading_info, quantity_precision)
        quantity = modify_order_quantity(quantity_precision, quantity)
        if (quantity > Decimal(0)) and ((latest_price * quantity) > 10):
            order = post_order(symbol, signal_type, quantity)
            trading_record(order, strategy, symbol, time_period, signal_type)
            processing_record(strategy, symbol, time_period, signal_type, order)
            status = order['status']
            orderId = order['orderId']
            avgPrice = order['avgPrice']
            executedQty = order['executedQty']
            rec01 = f'{status} Order : # {orderId} #'
            rec02 = f'{symbol}: {signal_type} Position {executedQty} at {avgPrice}'
            msg = rec01 + rec02
            send_message(msg)
        else:
            print('Order quantity is less than $10 or below the precision!'.center(120))
            print('Future Position Did Not Adjust Properly!'.center(120))
            a = 'Order quantity is less than $10 or below the precision!'
            b = 'Future Position Did Not Adjust Properly!'
            send_message(f'{a} {b}')


def post_order(symbol, signal_type, quantity):
    """
    发送订单, 处理交易所响应
    """
    order = \
        {
            'symbol': symbol,
            'side': binance_order_types[signal_type]['side'],
            'positionSide': binance_order_types[signal_type]['positionSide'],
            'quantity': quantity,
            'type': 'MARKET',
            'newOrderRespType': 'RESULT',
            'timestamp': int(time.time() * 1000)
        }
    order['quantity'] = str(order['quantity'])
    n = 0
    while True:
        try:
            order['timestamp'] = int(time.time() * 1000)
            order = exchange.fapiPrivatePostOrder(order)
            break
        except Exception:
            time.sleep(1)
            if n >= Max_atp:
                print('If you encounter difficulties, just don\'t do it and get a good night\'s sleep'.center(120))
                msg = 'Emergent Issue Occurred Boss, Please Check the Server!'
                send_message(msg)
                break
            else:
                n += 1
                continue
    status = order['status']
    orderId = order['orderId']
    avgPrice = order['avgPrice']
    executedQty = order['executedQty']
    rec01 = f'{status} Order : # {orderId} #'
    signal_type.replace('_', ' ')
    rec02 = f'{symbol}: {signal_type} Position {executedQty} at {avgPrice}'
    print('Order_Info'.center(120))
    print(f'{rec01}'.center(120))
    print(f' {rec02}'.center(120))
    return order


def intTodatetime(intValue):
    intValue = int(intValue)
    if len(str(intValue)) == 10:
        # 精确到秒
        timeValue = time.localtime(intValue)
        tempDate = time.strftime("%Y-%m-%d %H:%M:%S", timeValue)
        datetimeValue = datetime.strptime(tempDate, "%Y-%m-%d %H:%M:%S")
    elif 10 < len(str(intValue)) < 15:
        # 精确到毫秒
        k = len(str(intValue)) - 10
        timestamp = datetime.fromtimestamp(intValue / (1 * 10 ** k))
        datetimeValue = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        return -1
    return datetimeValue


def trading_record(order, strategy, symbol, time_period, signal_type):
    """
    目前功能暂时用于记录allocated_funds的变化, 通过获取的交易所响应, 计算当前订单的realized_PNL信息
    """
    quantity = order['executedQty']
    price = order['avgPrice']
    order_time = order['updateTime']
    side = signal_type
    order_time = intTodatetime(order_time)
    # record = pd.read_csv('data//trading_record.csv')
    record = pd.read_hdf(f'data//{strategy}_trading_record.h5', key=f'{symbol}', mode='a')
    record = pd.DataFrame(record).astype(str)
    df = \
        {
            'order_time': [f'{order_time}'],
            'strategy': [f'{strategy}'],
            'symbol': [f'{symbol}'],
            'time_period': [f'{time_period}'],
            'side': [f'{side}'],
            'Price': [f'{price}'],
            'quantity': [f'{quantity}'],
            'realized_PNL': ['didn\'t_calculated']
        }
    df = pd.DataFrame(df)
    df = record.append(df)
    df = df[~df['order_time'].isin(['insert_order_time'])]
    df['order_time'] = pd.to_datetime(df['order_time'])
    df.sort_values('order_time', inplace=True)
    df = df.astype(str)
    df.index = range(len(df))
    df.to_hdf(f'data//{strategy}_trading_record.h5', key=f'{symbol}', mode='a')
    # print(df)


def processing_record(strategy, symbol, time_period, signal_type, order):
    """
    通过record来计算PNL和allocated_funds
    """
    df = pd.read_hdf(f'data//{strategy}_trading_record.h5', key=f'{symbol}', mode='a')
    df = pd.DataFrame(df).astype(str)
    df_selected = df[df['time_period'].isin([f'{time_period}'])]
    df_selected['order_time'] = pd.to_datetime(df_selected['order_time'])
    df_selected.sort_values('order_time', inplace=True)
    df_unselected = df[~df['time_period'].isin([f'{time_period}'])]
    if signal_type in ['open_LONG', 'open_SHORT']:
        q = order['executedQty']
        p = order['avgPrice']
        n_funds = Decimal(q) * Decimal(p)
        trade_info = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
        trade_info = pd.DataFrame(trade_info).astype(str)
        o_funds = trade_info.loc[f'{time_period}', 'period_allocated_funds']
        o_funds = Decimal(o_funds)
        funds = o_funds - n_funds * Decimal('0.0004')
        funds = modify_decimal(funds)
        trade_info.loc[f'{time_period}', 'period_allocated_funds'] = funds
        trade_info = trade_info.astype(str)
        trade_info.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
        # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
        # print(df_02, strategy, symbol, time_period)
        # print('行号: ', str(sys._getframe().f_lineno))
        # print(df_selected.tail(1))
        df_selected.loc[df_selected.index[-1], 'realized_PNL'] = Decimal('0.000')
        df = df_unselected.append(df_selected)
        df['order_time'] = pd.to_datetime(df['order_time'])
        df.sort_values('order_time', inplace=True)
        df = df.astype(str)
        df.index = range(len(df))
        df.to_hdf(f'data//{strategy}_trading_record.h5', key=f'{symbol}', mode='a')
        # print(df)
    else:
        if signal_type in ['reduce_SHORT', 'close_SHORT']:
            trade_info = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
            trade_info = pd.DataFrame(trade_info).astype(str)
            side = Decimal(1)
            df_o = df_selected[df_selected[u'side'].isin(['open_SHORT'])]
            df_o['order_time'] = pd.to_datetime(df_o['order_time'])
            df_o.sort_values('order_time', inplace=True)
            # print(df_o)
        if signal_type in ['reduce_LONG', 'close_LONG']:
            trade_info = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
            trade_info = pd.DataFrame(trade_info).astype(str)
            side = Decimal(-1)
            df_o = df_selected[df_selected[u'side'].isin(['open_LONG'])]
            df_o['order_time'] = pd.to_datetime(df_o['order_time'])
            df_o.sort_values('order_time', inplace=True)
            # print(df_o)
        if len(df_selected.index) >= 2:
            n_record = df_selected.iloc[-1]
            # print(df_selected)
            o_record = df_o.iloc[-1]
            # print(df_o)
            n_funds = Decimal(n_record['quantity']) * Decimal(n_record['Price']) * Decimal(0.9996)
            o_funds = Decimal(n_record['quantity']) * Decimal(o_record['Price'])
            pnl = (n_funds - o_funds) * side
            # print(pnl, type(pnl))
            # pnl = Decimal(pnl.iloc[-1])
            # pnl = a.iloc[-1, a.columns.get_loc('a')]
            p = trade_info.loc[f'{time_period}', 'profit_trades']
            p = Decimal(p)
            gp = trade_info.loc[f'{time_period}', 'gross_profit']
            gp = Decimal(gp)
            l = trade_info.loc[f'{time_period}', 'loss_trades']
            l = Decimal(l)
            gl = trade_info.loc[f'{time_period}', 'gross_loss']
            gl = Decimal(gl)
            # print(pnl)
            if pnl >= 0:
                p += 1
                gp += pnl
                trade_info.loc[f'{time_period}', 'profit_trades'] = p
                trade_info.loc[f'{time_period}', 'gross_profit'] = gp
            else:
                l += 1
                gl += pnl
                trade_info.loc[f'{time_period}', 'loss_trades'] = l
                trade_info.loc[f'{time_period}', 'gross_loss'] = gl
            if (p+l) != 0 and gl != 0:
                trade_info.loc[f'{time_period}', 'profit_factor'] = Decimal(p/(p+l))
                trade_info.loc[f'{time_period}', 'profit_loss_ratio'] = Decimal(gp/abs(gl))
            n = trade_info.loc[f'{time_period}', 'period_allocated_funds']
            n = modify_decimal(n)
            n = n + pnl
            trade_info.loc[f'{time_period}', 'period_allocated_funds'] = Decimal(n)
            trade_info = trade_info.astype(str)
            df_selected.iloc[-1, df_selected.columns.get_loc('realized_PNL')] = pnl
            df = df_unselected.append(df_selected)
            df['order_time'] = pd.to_datetime(df['order_time'])
            df.sort_values('order_time', inplace=True)
            df = df.astype(str)
            df.index = range(len(df))
            df.to_hdf(f'data//{strategy}_trading_record.h5', key=f'{symbol}', mode='a')
            trade_info.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
            # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
            # print(df_02, strategy, symbol, time_period)
            # print('行号: ', str(sys._getframe().f_lineno))
        else:
            trade_info = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
            trade_info = pd.DataFrame(trade_info).astype(str)
            n_record = df_selected.tail(1)
            # print(df_selected)
            n_funds = n_record['quantity'].apply(Decimal) * n_record['Price'].apply(Decimal)
            o_funds = trade_info.loc[f'{time_period}', 'period_allocated_funds']
            o_funds = modify_decimal(o_funds)
            funds = o_funds - n_funds * Decimal('0.0004')
            n = modify_decimal(funds)
            trade_info.loc[f'{time_period}', 'period_allocated_funds'] = n
            trade_info = trade_info.astype(str)
            trade_info.to_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='a')
            # df_02 = pd.read_hdf(f'data//{strategy}.h5', key=f'{symbol}', mode='r')
            # print(df_02, strategy, symbol, time_period)
            # print('行号: ', str(sys._getframe().f_lineno))
            df_selected.iloc[-1, df_selected.columns.get_loc('realized_PNL')] = Decimal('0.000')
            df = df_unselected.append(df_selected)
            df['order_time'] = pd.to_datetime(df['order_time'])
            df.sort_values('order_time', inplace=True)
            df = df.astype(str)
            df.index = range(len(df))
            df.to_hdf(f'data//{strategy}_trading_record.h5', key=f'{symbol}', mode='a')
