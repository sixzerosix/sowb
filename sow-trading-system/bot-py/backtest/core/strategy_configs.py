import numpy as np
from backtest.core.config import StrategyConfig


# =========================================================================
# === 3 ВАРИАНТА СКАЛЬПИНГ-СТРАТЕГИЙ ===
# =========================================================================

# --- 1. АГРЕССИВНЫЙ ВЫСОКОЧАСТОТНЫЙ СКАЛЬПИНГ ---
# Цель: Максимальная частота сделок с очень узкими P/L.
# Используются быстрые индикаторы.
STRATEGY_CONFIG_HFT_SCALPER = StrategyConfig(
    initial_capital=100.0,
    leverage=20.0,
    target_roi_percent=0.8,  # Очень узкий тейк-профит
    risk_roi_percent=0.5,  # Очень узкий стоп-лосс
    indicator_set={
        "EMA_TREND": {"fast_len": 8, "slow_len": 18},  # Более быстрый тренд
        "ATR_EXIT": {
            "atr_len": 10,
            "min_multiplier": 1.5,
        },  # Более плотный SL на основе короткого ATR
        "SWING_STRUCT": {"window": 20},  # Быстрый свинг
        "HTF_FILTER": {"period": "15min", "ema_fast_len": 12, "ema_slow_len": 26},
        "FIBO": {
            "levels": np.array([0.236, 0.786], dtype=np.float64)
        },  # Более крайние уровни фибо
        "RSI": {"rsi_len": 5},  # Чрезвычайно быстрый RSI
        "MACD": {"fast_p": 8, "slow_p": 16, "signal_p": 5},
        "BOLLINGER_BANDS": {
            "period": 20,
            "dev_up": 1.8,
            "dev_dn": 1.8,
        },  # Узкие BB для чувствительности
        "VOLUME": {},
    },
)

# --- 2. СКАЛЬПИНГ НА ПРОБОЕ ВОЛАТИЛЬНОСТИ (BREAKOUT) ---
# Цель: Захват импульса после консолидации. Шире цели, шире риск.
STRATEGY_CONFIG_BREAKOUT_SCALPER = StrategyConfig(
    initial_capital=100.0,
    leverage=20.0,
    target_roi_percent=3.5,  # Умеренно широкий тейк-профит
    risk_roi_percent=2.0,  # Умеренно широкий стоп-лосс
    indicator_set={
        "EMA_TREND": {
            "fast_len": 14,
            "slow_len": 30,
        },  # Чуть медленнее тренд для фильтрации шума
        "ATR_EXIT": {
            "atr_len": 20,
            "min_multiplier": 3.0,
        },  # Более широкий SL на основе ATR для пробоя
        "SWING_STRUCT": {"window": 60},  # Более стабильная структура
        "HTF_FILTER": {
            "period": "30min",
            "ema_fast_len": 12,
            "ema_slow_len": 26,
        },  # Фильтр на более высоком ТФ
        "FIBO": {"levels": np.array([0.5, 0.707], dtype=np.float64)},
        "RSI": {
            "rsi_len": 14
        },  # Стандартный RSI для подтверждения перекупленности/перепроданности перед пробоем
        "MACD": {"fast_p": 12, "slow_p": 26, "signal_p": 9},
        "BOLLINGER_BANDS": {
            "period": 20,
            "dev_up": 3.0,
            "dev_dn": 3.0,
        },  # Широкие BB для обнаружения пробоя
        "VOLUME": {},
    },
)

# --- 3. СКАЛЬПИНГ НА ВОЗВРАТЕ К СРЕДНЕМУ (MEAN REVERSION) ---
# Цель: Торговля от границ канала с возвратом к скользящей средней (центру BB).
STRATEGY_CONFIG_MR_SCALPER = StrategyConfig(
    initial_capital=100.0,
    leverage=10.0,
    target_roi_percent=20,  # Тейк-профит у центра канала
    risk_roi_percent=10.0,  # Стоп-лосс за границей канала
    indicator_set={
        "EMA_TREND": {"fast_len": 20, "slow_len": 50},  # Медленный тренд для контекста
        "ATR_EXIT": {"atr_len": 20, "min_multiplier": 2.0},
        "SWING_STRUCT": {"window": 40},
        "HTF_FILTER": {"period": "30min", "ema_fast_len": 12, "ema_slow_len": 26},
        "FIBO": {"levels": np.array([0.382, 0.618], dtype=np.float64)},
        "RSI": {"rsi_len": 10},  # Средняя скорость RSI
        "MACD": {"fast_p": 12, "slow_p": 26, "signal_p": 9},
        "BOLLINGER_BANDS": {
            "period": 30,
            "dev_up": 2.0,
            "dev_dn": 2.0,
        },  # Более длинный период BB для более четкого канала
        "VOLUME": {},
    },
)
