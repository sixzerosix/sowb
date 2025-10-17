import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass, field
from typing import Dict, Any


# Mock-класс конфигурации для демонстрации
@dataclass
class StrategyConfig:
    """
    Конфигурация стратегии, содержащая набор индикаторов и их параметры.
    """

    indicator_set: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Здесь могут быть и другие параметры стратегии


def calculate_strategy_indicators(
    df: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """
    Расчет технических индикаторов на основе конфигурации StrategyConfig.indicator_set.

    :param df: Исходный DataFrame с OHLCV данными (должен содержать 'timestamp', 'open', 'high', 'low', 'close', 'volume').
    :param config: Объект StrategyConfig с настройками индикаторов.
    :return: DataFrame с добавленными колонками индикаторов.
    """
    # Создаем копию DataFrame, чтобы избежать SettingWithCopyWarning
    df = df.copy()
    indicators = config.indicator_set

    print(
        f"--- INFO: Начинаем расчет индикаторов. Исходное количество строк: {len(df)}"
    )

    # 1. EMAs (9 и 21) для определения тренда на 1m
    if "EMA_TREND" in indicators:
        params = indicators["EMA_TREND"]
        # Проверка наличия 'close' колонки уже выполнена в начале, но повторим для ясности
        if "close" in df.columns:
            df["ema_fast"] = talib.EMA(
                df["close"], timeperiod=params.get("fast_len", 9)
            )
            df["ema_slow"] = talib.EMA(
                df["close"], timeperiod=params.get("slow_len", 21)
            )
            print(
                f"--- DEBUG: Рассчитаны EMAs (fast={params.get('fast_len', 9)}, slow={params.get('slow_len', 21)})."
            )

    # 2. ATR для фильтрации входов/выходов
    if "ATR_EXIT" in indicators:
        params = indicators["ATR_EXIT"]
        df["atr_val"] = talib.ATR(
            df["high"], df["low"], df["close"], timeperiod=params.get("atr_len", 14)
        )
        print(f"--- DEBUG: Рассчитан ATR (len={params.get('atr_len', 14)}).")

    # 3. Расчет структурных точек (Swing High/Low)
    if "SWING_STRUCT" in indicators:
        params = indicators["SWING_STRUCT"]
        window = params.get("window", 5)
        # Rolling Window Center=True для определения структуры
        df["is_swing_high"] = (
            df["high"] == df["high"].rolling(window=window, center=True).max()
        )
        df["is_swing_low"] = (
            df["low"] == df["low"].rolling(window=window, center=True).min()
        )
        print(f"--- DEBUG: Рассчитаны структурные точки (window={window}).")

    # 4. Расчет HTF EMAs и их слияние (Merge with Higher Timeframe)
    if "HTF_FILTER" in indicators:
        params = indicators["HTF_FILTER"]

        # Агрегация данных в HTF (старший таймфрейм)
        df_htf = (
            df.set_index("timestamp")
            .resample(params.get("period", "30T"))  # "30T" for 30 minutes
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )

        # Расчет HTF EMAs
        df_htf["htf_ema_fast"] = talib.EMA(
            df_htf["close"], timeperiod=params.get("ema_fast_len", 10)
        )
        df_htf["htf_ema_slow"] = talib.EMA(
            df_htf["close"], timeperiod=params.get("ema_slow_len", 30)
        )

        df_htf["htf_is_uptrend"] = df_htf["htf_ema_fast"] > df_htf["htf_ema_slow"]

        df_htf_trend = df_htf[["timestamp", "htf_is_uptrend"]].rename(
            columns={"timestamp": "htf_timestamp"}
        )

        # merge_asof позволяет сопоставить 1m бар с последним завершенным HTF баром
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_htf_trend.sort_values("htf_timestamp"),
            left_on="timestamp",
            right_on="htf_timestamp",
            direction="backward",
        ).drop(columns=["htf_timestamp"])
        print(
            f"--- DEBUG: Рассчитан HTF Filter (period={params.get('period', '30T')})."
        )

    # 5. НОВЫЙ ИНДИКАТОР: RSI (Индекс относительной силы)
    if "RSI_FILTER" in indicators:
        params = indicators["RSI_FILTER"]
        df["rsi_val"] = talib.RSI(df["close"], timeperiod=params.get("rsi_len", 14))
        print(f"--- DEBUG: Рассчитан RSI (len={params.get('rsi_len', 14)}).")

    # Удаляем NaN после всех расчетов
    df_clean = df.dropna().reset_index(drop=True)

    print(
        f"--- INFO: После расчета индикаторов осталось {len(df_clean)} строк (удалено NaN)."
    )

    return df_clean


# --- Пример использования ---
# 1. Создание тестовых данных (1000 1-минутных свечей)
data = {
    "timestamp": pd.to_datetime(pd.date_range("2023-01-01", periods=1000, freq="1T")),
    "open": 100 + np.random.randn(1000).cumsum(),
    "high": 101 + np.random.randn(1000).cumsum(),
    "low": 99 + np.random.randn(1000).cumsum(),
    "close": 100 + np.random.randn(1000).cumsum(),
    "volume": np.random.randint(100, 1000, 1000),
}
df_test = pd.DataFrame(data)
df_test = pd.DataFrame(data)

# 2. Определение конфигурации стратегии
# Включаем EMA_TREND и наш новый RSI_FILTER, а Swing и HTF выключаем
strategy_config = StrategyConfig(
    indicator_set={
        "EMA_TREND": {
            "fast_len": 12,  # Переопределяем длину быстрой EMA
            "slow_len": 26,
        },
        "RSI_FILTER": {"rsi_len": 7},  # Используем короткий RSI
        "SWING_STRUCT": {
            # Этот индикатор не будет рассчитан, так как отсутствует в config.indicator_set
            # но если бы он был здесь, он бы использовал свои параметры
            "window": 3
        },
        # HTF_FILTER полностью выключен
    }
)

# 3. Расчет
df_with_indicators = calculate_strategy_indicators(df_test, strategy_config)

# 4. Вывод результата
print("\n--- Результат (Первые 5 строк) ---")
# Выводим только столбцы, которые нас интересуют: close, ema_fast, ema_slow, rsi_val
print(
    df_with_indicators[["timestamp", "close", "ema_fast", "ema_slow", "rsi_val"]].head()
)

# Проверка, что колонки HTF и SWING отсутствуют, так как они были выключены
print("\nКолонки в финальном DataFrame:")
print(list(df_with_indicators.columns))
