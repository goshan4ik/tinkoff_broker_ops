import time
import typing
from datetime import date

import finnhub
from enum import Enum


class FinnhubStocks(str, Enum):
    FACEBOOK = 'FB',
    APPLE = 'AAPL',
    AMAZON = 'AMZN',
    NETFLIX = 'NFLX',
    GOOGLE = 'GOOGL'


class FinnhubDataProvider:

    def __init__(self):
        self.__finnhub_client = finnhub.Client(api_key="c2n7csaad3i8g7sqr700")

    # TODO погрешность при расчетах??????
    def get_historical_data(self,
                            start_date: date,
                            end_date: date,
                            stocks: [FinnhubStocks]) -> typing.Dict[FinnhubStocks, typing.Dict[int, int]]:
        stocks_data = dict()
        start_timestamp = int(time.mktime(start_date.timetuple()))
        end_timestamp = int(time.mktime(end_date.timetuple()))
        for stock in stocks:
            res = self.__finnhub_client.stock_candles(stock, 'D', start_timestamp, end_timestamp)
            if res['s'] == 'ok':
                stock_data = dict()
                for i in range(0, len(res['t'])):
                    stock_data[res['t'][i]] = res['c'][i]
                stocks_data[stock] = stock_data
        return stocks_data
