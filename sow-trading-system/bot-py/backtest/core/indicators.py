import pandas as pd
import talib
import numpy as np
from typing import Dict, Any, Callable
from backtest.core.config import StrategyConfig  # Для типизации

# ----------------------------------------------------------------------
# ОТДЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАСЧЕТА КАЖДОГО ИНДИКАТОРА
# ----------------------------------------------------------------------


def _calculate_ema_trend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет EMAs (9 и 21) для определения тренда на 1m."""
    if "close" in df.columns:
        df["ema_fast"] = talib.EMA(df["close"], timeperiod=params.get("fast_len", 9))
        df["ema_slow"] = talib.EMA(df["close"], timeperiod=params.get("slow_len", 21))
        print(
            f"--- DEBUG: Рассчитаны EMAs (fast={params.get('fast_len', 9)}, slow={params.get('slow_len', 21)})."
        )
    return df


def _calculate_atr_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет ATR (Average True Range) для фильтрации входов/выходов."""
    df["atr_val"] = talib.ATR(
        df["high"], df["low"], df["close"], timeperiod=params.get("atr_len", 14)
    )
    # Здесь мы также можем добавить ATR-Exit-уровни для бэктеста
    df["atr_multiplier"] = params.get("min_multiplier", 2.0)
    print(f"--- DEBUG: Рассчитан ATR (len={params.get('atr_len', 14)}).")
    return df


def _calculate_swing_struct(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет структурных точек (Swing High/Low)."""
    window = params.get("window", 5)
    # Rolling Window Center=True для определения структуры
    df["is_swing_high"] = (
        df["high"] == df["high"].rolling(window=window, center=True).max()
    )
    df["is_swing_low"] = (
        df["low"] == df["low"].rolling(window=window, center=True).min()
    )
    print(f"--- DEBUG: Рассчитаны структурные точки (window={window}).")
    return df


def _calculate_htf_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет HTF EMAs и их слияние (Merge with Higher Timeframe)."""

    # ВАЖНО: Когда 'timestamp' установлен как индекс, нам нужно сначала сбросить его
    # для корректной работы set_index() и merge_asof()

    # 1. Сброс индекса, чтобы 'timestamp' стал колонкой
    df_reset = df.reset_index()

    # 2. Агрегация данных в HTF (старший таймфрейм)
    df_htf = (
        df_reset.set_index("timestamp")
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
    df_merged = pd.merge_asof(
        df_reset.sort_values("timestamp"),
        df_htf_trend.sort_values("htf_timestamp"),
        left_on="timestamp",
        right_on="htf_timestamp",
        direction="backward",
    ).drop(columns=["htf_timestamp"])

    # ВАЖНО: Восстанавливаем индекс времени перед возвратом
    df_merged = df_merged.set_index("timestamp")

    print(f"--- DEBUG: Рассчитан HTF Filter (period={params.get('period', '30T')}).")
    return df_merged


def _calculate_fibo(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Заглушка для Fibbonacci, так как логика его расчета сложна и
    зависит от Swing Structures, которые нужно найти ранее.
    Здесь мы просто добавляем нужные колонки (нужно для ядра Numba).
    """
    if "is_swing_high" in df.columns:
        # В реальной стратегии здесь была бы сложная логика поиска ближайших свингов
        # и расчета уровней Фибоначчи (0.382, 0.618 и т.д.)
        df["fibo_level_382"] = df["close"].shift(10)  # Mock data
        df["fibo_level_618"] = df["close"].shift(20)  # Mock data
        print(
            f"--- DEBUG: Добавлены Mock-колонки Фибоначчи (levels={params.get('levels', 'N/A')})."
        )
    else:
        print("--- WARNING: FIBO пропущен. Требуется SWING_STRUCT.")
    return df


# --- НОВЫЕ ИНДИКАТОРЫ ---


def _calculate_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет RSI (Индекс относительной силы)."""
    df["rsi_val"] = talib.RSI(df["close"], timeperiod=params.get("rsi_len", 14))
    print(f"--- DEBUG: Рассчитан RSI (len={params.get('rsi_len', 14)}).")
    return df


def _calculate_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет MACD (Схождение/расхождение скользящих средних)."""
    fast_p = params.get("fast_p", 12)
    slow_p = params.get("slow_p", 26)
    signal_p = params.get("signal_p", 9)

    macd, signal, hist = talib.MACD(
        df["close"], fastperiod=fast_p, slowperiod=slow_p, signalperiod=signal_p
    )
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist
    print(
        f"--- DEBUG: Рассчитан MACD (fast={fast_p}, slow={slow_p}, signal={signal_p})."
    )
    return df


def _calculate_parabolic_sar(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Parabolic SAR (Stop and Reverse)."""
    accel = params.get("accel", 0.02)
    max_accel = params.get("max_accel", 0.2)
    df["sar"] = talib.SAR(df["high"], df["low"], acceleration=accel, maximum=max_accel)
    print(f"--- DEBUG: Рассчитан Parabolic SAR (accel={accel}, max={max_accel}).")
    return df


def _calculate_bollinger_bands(
    df: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """Расчет Bollinger Bands (Полосы Боллинджера)."""
    period = params.get("period", 20)
    dev_up = params.get("dev_up", 2.0)
    dev_dn = params.get("dev_dn", 2.0)

    upper, middle, lower = talib.BBANDS(
        df["close"],
        timeperiod=period,
        nbdevup=dev_up,
        nbdevdn=dev_dn,
        matype=talib.MA_Type.SMA,
    )
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    print(f"--- DEBUG: Рассчитаны Bollinger Bands (period={period}, dev={dev_up}).")
    return df


def _calculate_stochastic(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Stochastic Oscillator (Стохастический осциллятор)."""
    fastk = params.get("fastk", 5)
    slowk = params.get("slowk", 3)
    slowd = params.get("slowd", 3)

    slowk_val, slowd_val = talib.STOCH(
        df["high"],
        df["low"],
        df["close"],
        fastk_period=fastk,
        slowk_period=slowk,
        slowk_matype=talib.MA_Type.SMA,
        slowd_period=slowd,
        slowd_matype=talib.MA_Type.SMA,
    )
    df["stoch_k"] = slowk_val
    df["stoch_d"] = slowd_val
    print(
        f"--- DEBUG: Рассчитан Stochastic Oscillator (fastk={fastk}, slowk={slowk}, slowd={slowd})."
    )
    return df


def _calculate_cci(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет CCI (Commodity Channel Index)."""
    period = params.get("period", 14)
    df["cci_val"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=period)
    print(f"--- DEBUG: Рассчитан CCI (period={period}).")
    return df


def _calculate_obv(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет OBV (On-Balance Volume)."""
    df["obv_val"] = talib.OBV(df["close"], df["volume"])
    print("--- DEBUG: Рассчитан OBV.")
    return df


def _calculate_standard_volume(
    df: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """Простой индикатор Объема (Volume)."""
    if "volume" in df.columns:
        # Просто убеждаемся, что колонка существует
        df["volume_val"] = df["volume"]
        print("--- DEBUG: Колонка Volume проверена и дублирована как volume_val.")
    else:
        print(
            "--- WARNING: Колонка 'volume' отсутствует в DataFrame. Индикатор Volume пропущен."
        )
    return df


# ----------------------------------------------------------------------
# ДИСПЕТЧЕР (КАРТА ФУНКЦИЙ)
# ----------------------------------------------------------------------
INDICATOR_DISPATCHER: Dict[
    str, Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
] = {
    # Основные индикаторы структуры и фильтрации
    "EMA_TREND": _calculate_ema_trend,
    "ATR_EXIT": _calculate_atr_exit,
    "SWING_STRUCT": _calculate_swing_struct,
    "HTF_FILTER": _calculate_htf_filter,
    "FIBO": _calculate_fibo,  # Добавлен FIBO
    # Запрошенные индикаторы тренда и осцилляторов
    "RSI": _calculate_rsi,
    "MACD": _calculate_macd,
    "PARABOLIC_SAR": _calculate_parabolic_sar,
    "BOLLINGER_BANDS": _calculate_bollinger_bands,
    "STOCHASTIC": _calculate_stochastic,
    "CCI": _calculate_cci,
    # Запрошенные индикаторы объёма
    "OBV": _calculate_obv,
    "VOLUME": _calculate_standard_volume,
}


# ----------------------------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ ОРКЕСТРАЦИИ
# ----------------------------------------------------------------------


def calculate_strategy_indicators(
    df: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """
    Расчет технических индикаторов на основе конфигурации StrategyConfig.indicator_set.
    """
    # Создаем копию DataFrame, чтобы избежать SettingWithCopyWarning
    df = df.copy()
    indicators = config.indicator_set

    print(
        f"--- INFO: Начинаем расчет индикаторов. Исходное количество строк: {len(df)}"
    )

    # Итерация по конфигурации и вызов соответствующих функций
    for indicator_name, params in indicators.items():
        if indicator_name in INDICATOR_DISPATCHER:
            try:
                # Динамический вызов функции расчета
                # ПРИМЕЧАНИЕ: Если _calculate_htf_filter возвращает df с reset_index,
                # это будет исправлено в конце этой функции, если мы не используем .reset_index()
                df = INDICATOR_DISPATCHER[indicator_name](df, params)
            except Exception as e:
                print(
                    f"--- ERROR: Не удалось рассчитать индикатор {indicator_name}: {e}"
                )
        else:
            print(
                f"--- WARNING: Пропущен неизвестный индикатор в конфигурации: {indicator_name}"
            )

    # Удаляем NaN после всех расчетов
    # ВАЖНОЕ ИСПРАВЛЕНИЕ: Удаляем .reset_index(drop=True), чтобы сохранить индекс времени!
    df_clean = df.dropna()

    print(
        f"--- INFO: После расчета индикаторов осталось {len(df_clean)} строк (удалено NaN)."
    )

    return df_clean
