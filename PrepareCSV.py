import pandas as pd
import datetime
import os


def prepare_data(time0, time9, symbol, fgCov=False, prep_new=True, mode='test'):
    path = 'D://Data//binance//futures//'
    df9path = f'..//data//{symbol}_1m_{mode}.csv'
    if prep_new:
        time0 = pd.to_datetime(time0)
        time0 = datetime.datetime.date(time0)
        time9 = pd.to_datetime(time9)
        time9 = datetime.datetime.date(time9)
        time5 = time0
        df9 = pd.DataFrame()
        while time5 <= time9:
            try:
                file = path + str(time5) + '//' + str(time5) + '_' + symbol + '_1m.csv'
                df0 = pd.read_csv(file)
                df0['datetime'] = [x[:19] for x in df0['candle_begin_time']]
                df0.set_index('datetime', drop=True, inplace=True)
                df0.index = pd.to_datetime(df0.index, format='%Y-%m-%d %H:%M:%S')
                df0.sort_index(ascending=True, inplace=True)
            except:
                time5 = time5 + datetime.timedelta(days=1)
                file = path + str(time5) + '//' + str(time5) + '_' + symbol + '_1m.csv'
                df0 = pd.read_csv(file)
                df0['datetime'] = [x[:19] for x in df0['candle_begin_time']]
                df0.set_index('datetime', drop=True, inplace=True)
                df0.index = pd.to_datetime(df0.index, format='%Y-%m-%d %H:%M:%S')
                df0.sort_index(ascending=True, inplace=True)
            df9 = df9.append(df0)
            df9.drop(columns=['candle_begin_time'], inplace=True)
            time5 = time5 + datetime.timedelta(days=1)
        # print(df9)
        # df9.reset_index(inplace=True)
        # df9['candle_begin_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # as_list = df9['candle_begin_time'].tolist()
        # for x in as_list:
        #     if '.000' in str(x):
        #         idx = as_list.index(x)
        #         as_list[idx] = idx[0:8]
        # df9.set_index(as_list, drop=True, inplace=True)

        df9.sort_index(ascending=True, inplace=True)
        df9.index = pd.to_datetime(df9.index, format='%Y-%m-%d %H:%M:%S')

        # df9 = df9.to_csv()
        if fgCov:
            df9 = df9
        else:
            df9.to_csv(df9path)
            df9 = df9path
        return df9
    else:
        df9 = df9path
        return df9


# if __name__ == '__main__':
#     time0, time9, symbol = '2018-01-01', '2020-01-01', 'ETHUSDT'
#     prepare_data(time0, time9, symbol, fgCov=False, prep_new=False)