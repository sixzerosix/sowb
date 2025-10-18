import pandas as pd
import talib
import numpy as np
from typing import Dict, Any, Callable
from backtest.core.config import StrategyConfig  # Для типизации

# ----------------------------------------------------------------------
# ОТДЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАСЧЕТА КАЖДОГО ИНДИКАТОРА
# ----------------------------------------------------------------------


def _calculate_ema_trend(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет EMAs для определения тренда."""
    if "close" in df.columns:
        fast_len = params.get("fast_len", 9)
        slow_len = params.get("slow_len", 21)
        df["ema_fast"] = talib.EMA(df["close"], timeperiod=fast_len)
        df["ema_slow"] = talib.EMA(df["close"], timeperiod=slow_len)
        print(f"--- DEBUG: Рассчитаны EMAs (fast={fast_len}, slow={slow_len}).")
    return df


def _calculate_atr_exit(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет ATR (Average True Range) для фильтрации входов/выходов."""
    atr_len = params.get("atr_len", 14)
    df["atr_val"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=atr_len)
    print(f"--- DEBUG: Рассчитан ATR (len={atr_len}).")
    return df


def _calculate_swing_struct(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Расчет структуры свингов (High/Low Points) с помощью скользящего окна.

    Определяет локальный пик, если текущая цена (High/Low) выше/ниже,
    чем в пределах N свечей до и после.
    """
    window = params.get("window", 10)  # Окно поиска свингов

    # Расчет, является ли текущая свеча самым высоким High в окне (local high)
    is_peak = (
        df["high"]
        == df["high"]
        .rolling(window=2 * window + 1, center=True, min_periods=window + 1)
        .max()
    )
    # Расчет, является ли текущая свеча самым низким Low в окне (local low)
    is_trough = (
        df["low"]
        == df["low"]
        .rolling(window=2 * window + 1, center=True, min_periods=window + 1)
        .min()
    )

    # Присваиваем значения только в точках пиков/впадин, остальное NaN
    df["swing_high"] = np.where(is_peak, df["high"], np.nan)
    df["swing_low"] = np.where(is_trough, df["low"], np.nan)

    # Заполняем NaN, чтобы иметь доступ к последнему свингу
    df["last_swing_high"] = df["swing_high"].ffill()
    df["last_swing_low"] = df["swing_low"].ffill()

    print(f"--- DEBUG: Рассчитаны SWING_STRUCT (window={window}).")
    return df


def _calculate_htf_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Фильтр по старшему таймфрейму (HTF).
    Агрегирует данные, рассчитывает EMA на HTF и проецирует тренд.
    """
    htf_period = params.get("period", "30min")  # Например, '30min', '1H', '4H'
    ema_len = params.get("ema_len", 20)

    # 1. Агрегация данных до HTF
    # Убедимся, что timestamp - это индекс типа DateTimeIndex для resample
    if not isinstance(df.index, pd.DatetimeIndex):
        # Создаем временную копию с DatetimeIndex, если 'timestamp' есть в колонках
        if "timestamp" in df.columns:
            temp_df = df.set_index("timestamp")
        else:
            print("--- ERROR: Не найдена колонка 'timestamp' для HTF расчета.")
            return df
    else:
        temp_df = df

    htf_df = (
        temp_df["close"]
        .resample(htf_period)
        .ohlc()  # Используем close HTF для простоты, но можно OHLV
        .dropna()  # Удаляем неполные HTF свечи
    )

    # 2. Расчет EMA на HTF
    htf_df["htf_ema"] = talib.EMA(htf_df["close"], timeperiod=ema_len)

    # 3. Реиндексация и слияние
    # Создаем DatetimeIndex на основе исходного DF для merge
    original_datetime_index = df["timestamp"] if "timestamp" in df.columns else df.index

    # Объединяем обратно с основным DF, используя ближайшее предыдущее значение
    df = pd.merge_asof(
        df.reset_index(drop=True),  # Сбрасываем индекс для merge_asof
        htf_df[["close", "htf_ema"]]
        .reset_index()
        .rename(columns={"index": "timestamp"}),
        on="timestamp",
        direction="backward",  # Берем последнее завершенное значение HTF
        suffixes=("", "_htf"),
    )

    # 4. Расчет тренда HTF
    df["htf_trend_up"] = df["close_htf"] > df["htf_ema"]

    print(f"--- DEBUG: Рассчитан HTF_FILTER (period={htf_period}, EMA={ema_len}).")
    return df


def _calculate_fibo(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Расчет уровня Фибоначчи.
    Предполагает, что SWING_STRUCT уже рассчитал 'last_swing_high' и 'last_swing_low'.
    """
    fibo_level = params.get("level", 0.618)  # Уровень 61.8% для коррекции

    # Для каждой строки рассчитываем коррекцию Фибоначчи от последнего значимого свинга
    swing_range = df["last_swing_high"] - df["last_swing_low"]

    # Расчет 61.8% коррекции от Low к High (для Long входа)
    df["fibo_level_long"] = df["last_swing_low"] + (swing_range * fibo_level)

    # Расчет 61.8% коррекции от High к Low (для Short входа)
    df["fibo_level_short"] = df["last_swing_high"] - (swing_range * fibo_level)

    print(f"--- DEBUG: Рассчитаны FIBO уровни (level={fibo_level}).")
    return df


# --- Индикаторы Тренда и Осцилляторы (Используется TA-Lib) ---


def _calculate_rsi(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Индекса Относительной Силы (RSI)."""
    rsi_len = params.get("rsi_len", 14)
    df["rsi_val"] = talib.RSI(df["close"], timeperiod=rsi_len)
    print(f"--- DEBUG: Рассчитан RSI (len={rsi_len}).")
    return df


def _calculate_macd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Схождения/Расхождения Скользящих Средних (MACD)."""
    fast = params.get("fast_len", 12)
    slow = params.get("slow_len", 26)
    signal = params.get("signal_len", 9)
    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"],
    ) = talib.MACD(df["close"], fastperiod=fast, slowperiod=slow, signalperiod=signal)
    print(f"--- DEBUG: Рассчитан MACD (fast={fast}, slow={slow}, signal={signal}).")
    return df


def _calculate_parabolic_sar(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Parabolic SAR (Stop and Reverse)."""
    accel = params.get("acceleration", 0.02)
    limit = params.get("max_acceleration", 0.2)
    df["psar"] = talib.SAR(df["high"], df["low"], acceleration=accel, maximum=limit)
    print(f"--- DEBUG: Рассчитан PSAR (accel={accel}, max={limit}).")
    return df


def _calculate_bollinger_bands(
    df: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """Расчет Полос Боллинджера (Bollinger Bands)."""
    bb_len = params.get("bb_len", 20)
    num_dev = params.get("num_dev", 2.0)
    (
        df["bb_upper"],
        df["bb_middle"],
        df["bb_lower"],
    ) = talib.BBANDS(
        df["close"],
        timeperiod=bb_len,
        nbdevup=num_dev,
        nbdevdn=num_dev,
        matype=talib.MA_Type.SMA,
    )
    print(f"--- DEBUG: Рассчитаны Bollinger Bands (len={bb_len}, dev={num_dev}).")
    return df


def _calculate_stochastic(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Стохастического Осциллятора (Stochastic Oscillator)."""
    k_len = params.get("k_len", 14)
    d_len = params.get("d_len", 3)
    (
        df["stoch_k"],
        df["stoch_d"],
    ) = talib.STOCH(
        df["high"],
        df["low"],
        df["close"],
        fastk_period=k_len,
        slowk_period=d_len,
        slowd_period=d_len,
        slowk_matype=talib.MA_Type.SMA,
        slowd_matype=talib.MA_Type.SMA,
    )
    print(f"--- DEBUG: Рассчитан Stochastic (K={k_len}, D={d_len}).")
    return df


def _calculate_cci(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Индекса Товарного Канала (Commodity Channel Index - CCI)."""
    cci_len = params.get("cci_len", 14)
    df["cci_val"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=cci_len)
    print(f"--- DEBUG: Рассчитан CCI (len={cci_len}).")
    return df


# --- Индикаторы Объёма ---


def _calculate_obv(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Расчет Балансового Объема (On-Balance Volume - OBV)."""
    # Проверяем наличие 'volume', так как OBV использует его
    if "volume" in df.columns:
        df["obv_val"] = talib.OBV(df["close"], df["volume"])
        print("--- DEBUG: Рассчитан OBV.")
    else:
        df["obv_val"] = 0
        print("--- WARNING: Колонка 'volume' отсутствует. OBV установлен в 0.")
    return df


def _calculate_standard_volume(
    df: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """Простое включение колонки объема в набор индикаторов."""
    if "volume" in df.columns:
        df["std_volume"] = df["volume"]
        print("--- DEBUG: Добавлен стандартный VOLUME.")
    else:
        df["std_volume"] = 0
        print("--- WARNING: Колонка 'volume' отсутствует. Добавлен 0-объем.")
    return df


# ----------------------------------------------------------------------
# ДИСПЕТЧЕР ИНДИКАТОРОВ
# ----------------------------------------------------------------------

INDICATOR_DISPATCHER: Dict[
    str, Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
] = {
    # Основные индикаторы структуры и фильтрации
    "EMA_TREND": _calculate_ema_trend,
    "ATR_EXIT": _calculate_atr_exit,
    "SWING_STRUCT": _calculate_swing_struct,
    "HTF_FILTER": _calculate_htf_filter,
    "FIBO": _calculate_fibo,
    # Индикаторы тренда и осцилляторов
    "RSI": _calculate_rsi,
    "MACD": _calculate_macd,
    "PARABOLIC_SAR": _calculate_parabolic_sar,
    "BOLLINGER_BANDS": _calculate_bollinger_bands,
    "STOCHASTIC": _calculate_stochastic,
    "CCI": _calculate_cci,
    # Индикаторы объёма
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

    # 1. Применяем timestamp как DatetimeIndex, если его еще нет
    # Это важно для resample в HTF_FILTER
    if "timestamp" in df.columns:
        df = df.set_index("timestamp", drop=False)

    # Конвертируем индекс в DatetimeIndex, если это еще не так
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            # Если преобразование не удалось, полагаемся на колонку 'timestamp'
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp", drop=False)
            else:
                print("--- ERROR: Невозможно определить временной индекс для расчета.")
                return df

    indicators = config.indicator_set

    print(
        f"--- INFO: Начинаем расчет индикаторов. Исходное количество строк: {len(df)}"
    )

    # Итерация по конфигурации и вызов соответствующих функций
    for indicator_name, params in indicators.items():
        if indicator_name in INDICATOR_DISPATCHER:
            try:
                # Динамический вызов функции расчета
                df = INDICATOR_DISPATCHER[indicator_name](df, params)
            except Exception as e:
                print(
                    f"--- ERROR: Не удалось расsсчитать индикатор {indicator_name}: {e}"
                )
        else:
            print(
                f"--- WARNING: Пропущен неизвестный индикатор в конфигурации: {indicator_name}"
            )

    # Удаляем NaN после всех расчетов
    # ВАЖНО: dropna() удалит строки с NaN, возникшие из-за периода расчета (например, 21 свеча для EMA)
    df_clean = df.dropna().reset_index(drop=True)

    print(
        f"--- INFO: Расчет индикаторов завершен. Строк после очистки NaN: {len(df_clean)}"
    )
    return df_clean
