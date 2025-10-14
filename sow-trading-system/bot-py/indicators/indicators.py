import pandas as pd
import talib


def calculate_indicators(df):
    # EMA
    df["ema9"] = talib.EMA(df["close"], timeperiod=9)
    df["ema21"] = talib.EMA(df["close"], timeperiod=21)

    # Bollinger Bands
    period_bb = 9
    df["basisBB"] = talib.SMA(df["close"], timeperiod=period_bb)
    df["stdBB"] = df["close"].rolling(window=period_bb).std()
    df["upperBB"] = df["basisBB"] + 2 * df["stdBB"]
    df["lowerBB"] = df["basisBB"] - 2 * df["stdBB"]

    # VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_vp"] = (tp * df["volume"]).cumsum()
    df["vwap"] = df["cum_vp"] / df["cum_vol"]

    # RSI
    df["rsi5"] = talib.RSI(df["close"], timeperiod=5)
    df["rsi14"] = talib.RSI(df["close"], timeperiod=14)

    # Fibonacci Levels (расчёт по последним 100 барам)
    if len(df) >= 100:
        last_n = df[-100:]
    else:
        last_n = df
    highFib = last_n["high"].max()
    lowFib = last_n["low"].min()
    diffFib = highFib - lowFib
    fib_levels = {
        "fib_0": lowFib,
        "fib_236": lowFib + diffFib * 0.236,
        "fib_382": lowFib + diffFib * 0.382,
        "fib_50": lowFib + diffFib * 0.5,
        "fib_618": lowFib + diffFib * 0.618,
        "fib_786": lowFib + diffFib * 0.786,
        "fib_1": highFib,
    }
    df["fib_levels"] = [fib_levels] * len(df)

    # OBV и Delta
    df["obv"] = talib.OBV(df["close"], df["volume"])
    df["delta"] = df["obv"].diff()

    return df


def check_signals(df):
    df["prev_ema9"] = df["ema9"].shift(1)
    df["prev_ema21"] = df["ema21"].shift(1)

    ema_cross_up = (df["ema9"] > df["ema21"]) & (df["prev_ema9"] <= df["prev_ema21"])
    ema_cross_down = (df["ema9"] < df["ema21"]) & (df["prev_ema9"] >= df["prev_ema21"])

    latest_cross_up = ema_cross_up.iloc[-1]
    latest_cross_down = ema_cross_down.iloc[-1]

    return latest_cross_up, latest_cross_down
