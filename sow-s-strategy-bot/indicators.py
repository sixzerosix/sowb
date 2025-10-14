import pandas as pd
import numpy as np
import talib
from typing import Dict, List


class TechnicalIndicators:
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        return talib.EMA(df["close"], timeperiod=period)

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI индикатор"""
        return talib.RSI(df["close"], timeperiod=period)

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX индикатор силы тренда"""
        return talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        return talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, period: int = 20, std: float = 2
    ) -> Dict:
        """Полосы Боллинджера"""
        upper, middle, lower = talib.BBANDS(
            df["close"], timeperiod=period, nbdevup=std, nbdevdn=std
        )
        return {"bb_upper": upper, "bb_middle": middle, "bb_lower": lower}

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_vol_price = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        return cumulative_vol_price / cumulative_vol

    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> Dict:
        """Уровни Фибоначчи"""
        recent_data = df.tail(lookback)
        high = recent_data["high"].max()
        low = recent_data["low"].min()

        diff = high - low
        levels = {
            "fib_0": high,
            "fib_236": high - 0.236 * diff,
            "fib_382": high - 0.382 * diff,
            "fib_50": high - 0.5 * diff,
            "fib_618": high - 0.618 * diff,
            "fib_786": high - 0.786 * diff,
            "fib_100": low,
        }
        return levels

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        return talib.OBV(df["close"], df["volume"])

    @staticmethod
    def add_all_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
        """Добавление всех индикаторов к DataFrame"""
        if len(df) < 50:
            return df

        # EMA
        df["ema_9"] = TechnicalIndicators.calculate_ema(df, config.EMA_FAST)
        df["ema_21"] = TechnicalIndicators.calculate_ema(df, config.EMA_SLOW)

        # RSI
        df["rsi_5"] = TechnicalIndicators.calculate_rsi(df, config.RSI_PERIOD_SHORT)
        df["rsi_14"] = TechnicalIndicators.calculate_rsi(df, config.RSI_PERIOD_LONG)

        # ADX
        df["adx"] = TechnicalIndicators.calculate_adx(df)

        # ATR
        df["atr"] = TechnicalIndicators.calculate_atr(df)

        # Bollinger Bands
        bb = TechnicalIndicators.calculate_bollinger_bands(
            df, config.BB_PERIOD, config.BB_STD
        )
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]

        # VWAP
        df["vwap"] = TechnicalIndicators.calculate_vwap(df)

        # OBV
        df["obv"] = TechnicalIndicators.calculate_obv(df)

        return df
