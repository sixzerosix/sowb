import os


def market_data():
    return [{"symbol": "BTC/USDT", "price": 96000}]


def risk():
    print("Risk system is watching...")


def exchange():
    return ["ByBit"]


def triggers(pool: list):
    pass


def signals_pool():
    pass


def signal(nane: str, options: dict):
    pass


def stratery(indicators: list["Indicator"]):
    pass


def indicator(params: dict):
    pass


# ______________________________________________


class Indicator(object):
    def __init__(self, name: str):
        self.name = name

    def EMA(data: list):
        pass


Ema = Indicator("EMA").EMA()
