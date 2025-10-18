"""
ИСПРАВЛЕННАЯ ВЕРСИЯ СИСТЕМЫ БЭКТЕСТИНГА v4

ОСНОВНЫЕ ИСПРАВЛЕНИЯ:
1. Исправлена ошибка Numba с динамическим списком (используем предварительную аллокацию)
2. Добавлен расчет Sharpe Ratio
3. Исправлены вызовы функций с правильными параметрами
4. Улучшена архитектура и читаемость кода
"""

import time
import pandas as pd
import numpy as np
import itertools
import copy
from typing import Tuple, Dict, Any
from numba import njit

from rich.console import Console
import matplotlib.pyplot as plt

# Импорты модулей
from backtest.core.config import StrategyConfig, PersistenceConfig, FILE_PATH
from backtest.core.data_loader import load_data
from backtest.core.indicators import calculate_strategy_indicators
from backtest.core.analysis import calculate_metrics, display_results_rich, plot_results
from backtest.core.persistence import (
    persist_results,
    persist_optimization_result,
    SCRIPT_DIR,
)

console = Console()

# =========================================================================
# === ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ПЕРЕКЛЮЧАТЕЛИ РЕЖИМОВ ===
# =========================================================================

# --- ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМА ---
RUN_MODE = "OPTIMIZE"  # "SINGLE" или "OPTIMIZE"

# Какую стратегию запускать в режиме "SINGLE"
TARGET_CONFIG_NAME = "EMA_RSI_ATR_Strategy"

# --- МЕТРИКА ОПТИМИЗАЦИИ ---
OPTIMIZATION_METRIC = "Sharpe Ratio"  # Доступные: "Total PnL", "Return (%)", "Sharpe Ratio", "Max Drawdown (%)"

# =========================================================================
# === КОНФИГУРАЦИЯ СТРАТЕГИЙ И ПАРАМЕТРОВ ОПТИМИЗАЦИИ ===
# =========================================================================

# --- 1. ПАРАМЕТРИЧЕСКОЕ ПРОСТРАНСТВО ДЛЯ ОПТИМИЗАЦИИ ---
PARAMETER_SPACE = {
    "EMA_TREND": {
        "fast_len": [5, 9, 13],
        "slow_len": [21, 34, 50],
    },
    "RSI": {
        "rsi_len": [14, 20],
    },
    "EXIT": {
        "target_roi_percent": [0.5, 1.0],
        "risk_roi_percent": [0.5, 0.8],
    },
}

# --- 2. КОНФИГУРАЦИЯ СТРАТЕГИЙ ---
STRATEGY_CONFIGS: Dict[str, StrategyConfig] = {
    "EMA_RSI_ATR_Strategy": StrategyConfig(
        initial_capital=100.0,
        leverage=10.0,
        target_roi_percent=20.0,
        risk_roi_percent=10.0,
        indicator_set={
            "EMA_TREND": {"fast_len": 9, "slow_len": 21},
            "RSI": {"rsi_len": 14, "overbought": 70, "oversold": 30},
            "ATR_EXIT": {"atr_len": 14, "atr_multiplier": 1.5},
        },
    ),
}

# =========================================================================
# === МОДУЛЬ 4: NUMBA ЯДРО БЭКТЕСТА (@njit) - ИСПРАВЛЕННАЯ ВЕРСИЯ ===
# =========================================================================


@njit
def _numba_backtest_core(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    rsi: np.ndarray,
    leverage: float,
    target_roi_perc: float,
    risk_roi_perc: float,
) -> Tuple[np.ndarray, float]:
    """
    ИСПРАВЛЕННОЕ ядро бэктеста с предварительной аллокацией массива сделок.

    КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
    Вместо динамического списка используем предварительно выделенный 2D массив.
    Это решает проблему Numba с list(array(...)).
    """

    num_bars = len(close_prices)

    # ИСПРАВЛЕНИЕ: Предварительно выделяем массив для максимально возможного числа сделок
    # В худшем случае - одна сделка на каждую свечу
    max_trades = num_bars
    trades_array = np.zeros((max_trades, 8), dtype=np.float64)
    trade_count = 0

    # Состояние позиций
    is_in_position = False
    side = 0  # 1: Long, -1: Short, 0: None
    entry_index = -1
    entry_price = 0.0

    # Работаем с капиталом как с множителем (начинаем с 1.0)
    current_equity = 1.0

    # Преобразуем проценты в доли
    target_roi = target_roi_perc / 100.0
    risk_roi = risk_roi_perc / 100.0

    # Параметры сделки
    tp_price = 0.0
    sl_price = 0.0
    position_size_perc = 0.10  # 10% капитала на сделку

    # ----------------------------------------------------------------------
    # --- Основной цикл бэктеста ---
    # ----------------------------------------------------------------------
    for i in range(1, num_bars):
        current_close = close_prices[i]

        # 1. ОБРАБОТКА ОТКРЫТОЙ ПОЗИЦИИ (Exit Logic)
        is_exit = False
        if is_in_position:
            pnl_perc = 0.0
            liquidation = 0

            # Проверка выхода Long (TP/SL)
            if side == 1:
                if high_prices[i] >= tp_price:
                    exit_price = tp_price
                    pnl_perc = ((exit_price / entry_price) - 1.0) * leverage
                    is_exit = True
                elif low_prices[i] <= sl_price:
                    exit_price = sl_price
                    pnl_perc = ((exit_price / entry_price) - 1.0) * leverage
                    is_exit = True

            # Проверка выхода Short (TP/SL)
            elif side == -1:
                if low_prices[i] <= tp_price:
                    exit_price = tp_price
                    pnl_perc = (1.0 - (exit_price / entry_price)) * leverage
                    is_exit = True
                elif high_prices[i] >= sl_price:
                    exit_price = sl_price
                    pnl_perc = (1.0 - (exit_price / entry_price)) * leverage
                    is_exit = True

            if is_exit:
                # Обновляем эквити
                current_equity *= 1.0 + pnl_perc

                volume_usd_ratio = position_size_perc * leverage

                # ИСПРАВЛЕНИЕ: Записываем в предварительно выделенный массив
                trades_array[trade_count, 0] = float(i)  # exit_index
                trades_array[trade_count, 1] = float(entry_index)
                trades_array[trade_count, 2] = float(side)
                trades_array[trade_count, 3] = entry_price
                trades_array[trade_count, 4] = exit_price
                trades_array[trade_count, 5] = pnl_perc
                trades_array[trade_count, 6] = volume_usd_ratio
                trades_array[trade_count, 7] = float(liquidation)

                trade_count += 1

                # Сброс состояния
                is_in_position = False
                side = 0
                entry_index = -1
                entry_price = 0.0

        # 2. ЛОГИКА ВХОДА (Entry Logic)
        if not is_in_position:
            # Расчет сигналов на основе индикаторов предыдущей свечи (i-1)
            long_signal = (ema_fast[i - 1] > ema_slow[i - 1]) and (rsi[i - 1] < 30)
            short_signal = (ema_fast[i - 1] < ema_slow[i - 1]) and (rsi[i - 1] > 70)

            # Вход Long
            if long_signal:
                is_in_position = True
                side = 1
                entry_index = i
                entry_price = current_close

                tp_price = entry_price * (1.0 + target_roi)
                sl_price = entry_price * (1.0 - risk_roi)

            # Вход Short
            elif short_signal:
                is_in_position = True
                side = -1
                entry_index = i
                entry_price = current_close

                tp_price = entry_price * (1.0 - target_roi)
                sl_price = entry_price * (1.0 + risk_roi)

    # После цикла: Если позиция осталась открытой, закрываем ее по последней цене
    if is_in_position:
        exit_price = close_prices[-1]
        pnl_perc = 0.0

        if side == 1:
            pnl_perc = ((exit_price / entry_price) - 1.0) * leverage
        elif side == -1:
            pnl_perc = (1.0 - (exit_price / entry_price)) * leverage

        current_equity *= 1.0 + pnl_perc

        volume_usd_ratio = position_size_perc * leverage

        trades_array[trade_count, 0] = float(num_bars - 1)
        trades_array[trade_count, 1] = float(entry_index)
        trades_array[trade_count, 2] = float(side)
        trades_array[trade_count, 3] = entry_price
        trades_array[trade_count, 4] = exit_price
        trades_array[trade_count, 5] = pnl_perc
        trades_array[trade_count, 6] = volume_usd_ratio
        trades_array[trade_count, 7] = 0.0

        trade_count += 1

    # ИСПРАВЛЕНИЕ: Возвращаем только заполненную часть массива
    return trades_array[:trade_count], current_equity


# =========================================================================
# === МОДУЛЬ 5: ДВИГАТЕЛЬ БЭКТЕСТА (Python Wrapper) ===
# =========================================================================


def backtest_engine(
    df: pd.DataFrame, config: StrategyConfig
) -> Tuple[pd.DataFrame, float]:
    """
    Основной двигатель бэктеста. Подготавливает данные и вызывает ядро Numba.
    """

    # 1. Подготовка данных для Numba
    close_prices = df["close"].values.astype(np.float64)
    high_prices = df["high"].values.astype(np.float64)
    low_prices = df["low"].values.astype(np.float64)

    ema_fast = df["ema_fast"].values.astype(np.float64)
    ema_slow = df["ema_slow"].values.astype(np.float64)
    rsi = df["rsi_val"].values.astype(np.float64)

    # 2. Вызов ядра Numba
    trades_array, final_equity_ratio = _numba_backtest_core(
        close_prices,
        high_prices,
        low_prices,
        ema_fast,
        ema_slow,
        rsi,
        config.leverage,
        config.target_roi_percent,
        config.risk_roi_percent,
    )

    # 3. Преобразование результатов Numba обратно в DataFrame
    if trades_array.size == 0:
        print("--- WARNING: Numba ядро не вернуло ни одной сделки. ---")
        return pd.DataFrame(), config.initial_capital

    trades_df = pd.DataFrame(
        trades_array,
        columns=[
            "exit_index",
            "entry_index",
            "side",
            "entry_price",
            "exit_price",
            "pnl_perc",
            "volume_usd_ratio",
            "liquidation",
        ],
    )

    # Преобразование индексов в целые числа
    trades_df["exit_index"] = trades_df["exit_index"].astype(int)
    trades_df["entry_index"] = trades_df["entry_index"].astype(int)

    # Сопоставление индексов с временными метками
    trades_df["entry_time"] = df.loc[trades_df["entry_index"], "timestamp"].values
    trades_df["exit_time"] = df.loc[trades_df["exit_index"], "timestamp"].values

    # Расчет PNL в абсолютных единицах
    trade_volume_usd = config.initial_capital * trades_df["volume_usd_ratio"]
    trades_df["pnl"] = trades_df["pnl_perc"] * trade_volume_usd

    # Расчет эквити после каждой сделки
    trades_df["equity_after_trade"] = config.initial_capital + trades_df["pnl"].cumsum()

    # Обновление финальной эквити
    final_equity = config.initial_capital * final_equity_ratio

    # Удаляем служебный столбец
    trades_df = trades_df.drop(columns=["volume_usd_ratio"])

    return trades_df, final_equity


# =========================================================================
# === МОДУЛЬ 3: ОРКЕСТРАТОР ПРОГОНА (RUNNER) ===
# =========================================================================


def run_backtest(
    config: StrategyConfig, data_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict, float]:
    """
    Выполняет полный цикл бэктеста: расчет индикаторов -> ядро бэктеста -> анализ.
    """

    start_time = time.time()

    # 1. Расчет индикаторов
    df_with_indicators = calculate_strategy_indicators(data_df, config)

    if df_with_indicators.empty:
        print("--- ERROR: DataFrame пуст после расчета индикаторов и удаления NaN. ---")
        return pd.DataFrame(), {}, time.time() - start_time

    # 2. Выполнение бэктеста
    trades_df, final_equity = backtest_engine(df_with_indicators, config)

    # 3. Анализ результатов
    metrics, drawdown, equity_curve = calculate_metrics(
        trades_df, config.initial_capital, final_equity
    )

    execution_time = time.time() - start_time

    return trades_df, metrics, execution_time


# =========================================================================
# === РЕФАКТОРИНГ: ОТДЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РЕЖИМОВ ===
# =========================================================================


def run_single_backtest(
    data_df: pd.DataFrame, config: StrategyConfig, persistence_config: PersistenceConfig
):
    """Выполняет и отображает результаты одного прогона стратегии."""
    print(f"\n--- РЕЖИМ: ОДИНОЧНЫЙ ПРОГОН ({TARGET_CONFIG_NAME}) ---")

    trades_df, metrics, execution_time = run_backtest(config, data_df)

    print("\n--- РЕЗУЛЬТАТЫ ОДИНОЧНОГО ПРОГОНА ---")
    display_results_rich(metrics, trades_df, execution_time)

    if not trades_df.empty:
        print("\n--- ГРАФИК И СОХРАНЕНИЕ ---")

        # ИСПРАВЛЕНИЕ: Передаем все необходимые параметры
        _, drawdown, equity_curve = calculate_metrics(
            trades_df,
            config.initial_capital,
            config.initial_capital + trades_df["pnl"].sum(),
        )

        plot_results(trades_df, config.initial_capital, equity_curve, drawdown)
        persist_results(trades_df, persistence_config)
    else:
        print("\n--- WARNING: В одиночном прогоне не совершено ни одной сделки. ---")


def run_optimization(
    data_df: pd.DataFrame,
    base_config: StrategyConfig,
    parameter_space: Dict,
    optimization_metric: str,
    persistence_config: PersistenceConfig,
):
    """Выполняет цикл оптимизации и сохраняет лучший результат."""
    print("\n--- РЕЖИМ: ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ---")
    print(f"--- ЦЕЛЬ: Максимизировать '{optimization_metric}' ---")

    # 1. Генерация всех комбинаций параметров
    param_keys = []
    param_values = []

    for ind_name, ind_params in parameter_space.items():
        if ind_name in base_config.indicator_set or ind_name == "EXIT":
            for param_key, values in ind_params.items():
                param_keys.append((ind_name, param_key))
                param_values.append(values)

    param_combinations = itertools.product(*param_values)

    total_runs = np.prod([len(v) for v in param_values])
    if total_runs == 0:
        print("--- WARNING: Не найдено параметров для оптимизации. ---")
        return

    print(f"--- INFO: Всего комбинаций для оптимизации: {int(total_runs)} ---")

    best_metric_value = -np.inf
    best_config = None
    best_metrics = None
    best_trades_df = pd.DataFrame()

    optimization_start_time = time.time()

    run_count = 0

    # 2. Основной цикл оптимизации
    for combination in param_combinations:
        run_count += 1
        current_config = copy.deepcopy(base_config)

        # Применение текущей комбинации к конфигурации
        for (ind_name, param_key), value in zip(param_keys, combination):
            if ind_name == "EXIT":
                if param_key == "target_roi_percent":
                    current_config.target_roi_percent = value
                elif param_key == "risk_roi_percent":
                    current_config.risk_roi_percent = value
            else:
                if ind_name not in current_config.indicator_set:
                    current_config.indicator_set[ind_name] = {}
                current_config.indicator_set[ind_name][param_key] = value

        # Проверка логичности параметров (fast_len < slow_len)
        ema_fast_len = current_config.indicator_set.get("EMA_TREND", {}).get(
            "fast_len", 0
        )
        ema_slow_len = current_config.indicator_set.get("EMA_TREND", {}).get(
            "slow_len", 0
        )

        if ema_fast_len >= ema_slow_len and ema_fast_len != 0:
            continue

        # Запуск бэктеста
        current_trades_df, metrics, _ = run_backtest(current_config, data_df)

        # 3. Оценка результата
        current_metric_value = metrics.get(optimization_metric, -np.inf)

        # Обновление лучшего результата
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_config = current_config
            best_metrics = metrics
            best_trades_df = current_trades_df

    optimization_end_time = time.time()
    execution_time_opt = optimization_end_time - optimization_start_time

    # 4. Вывод и сохранение лучших результатов
    if best_config and best_metrics:
        print("\n--- РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ (ЛУЧШИЙ ПРОГОН) ---")
        print(f"Общее время оптимизации: {execution_time_opt:.2f} сек.")
        print(f"Проверено прогонов: {run_count}")

        print("\nПараметры лучшей конфигурации:")
        for ind_name, ind_params in parameter_space.items():
            if ind_name == "EXIT":
                for param_key in ind_params.keys():
                    value = getattr(best_config, param_key)
                    print(f"- {param_key}: {value}")
            elif ind_name in best_config.indicator_set:
                for param_key in ind_params.keys():
                    value = best_config.indicator_set[ind_name].get(param_key, "N/A")
                    print(f"- {ind_name} {param_key}: {value}")

        display_results_rich(best_metrics, best_trades_df, execution_time_opt)

        if not best_trades_df.empty:
            print("\n--- ГРАФИК ЛУЧШЕГО РЕЗУЛЬТАТА ---")
            _, drawdown, equity_curve = calculate_metrics(
                best_trades_df,
                best_config.initial_capital,
                best_config.initial_capital + best_trades_df["pnl"].sum(),
            )
            plot_results(
                best_trades_df, best_config.initial_capital, equity_curve, drawdown
            )
        else:
            print(
                "\n--- WARNING: Нет сделок для построения графика в лучшем прогоне. ---"
            )

        print("\n--- СОХРАНЕНИЕ ЛУЧШЕГО РЕЗУЛЬТАТА ---")
        persist_optimization_result(best_config, best_metrics, persistence_config)
    else:
        print("\n--- ОШИБКА ОПТИМИЗАЦИИ ---")
        print("Не удалось найти лучшую конфигурацию.")


# =========================================================================
# === ОСНОВНАЯ ФУНКЦИЯ (MAIN) ===
# =========================================================================


def main():
    """
    Основная функция для запуска бэктеста или оптимизации.
    """
    print("--- ЗАПУСК СИСТЕМЫ БЭКТЕСТА ---")

    # 1. Загрузка данных
    data_df = load_data(str(FILE_PATH))

    if data_df.empty:
        print("--- ERROR: Не удалось загрузить данные. Выход. ---")
        return

    # --- КОНФИГУРАЦИЯ СОХРАНЕНИЯ (Общая) ---
    persistence_config = PersistenceConfig(
        save_to_sqlite=False,
        save_to_csv=True,
        save_to_txt=False,
        table_name="backtest_trades",
    )

    if RUN_MODE == "SINGLE":
        config = STRATEGY_CONFIGS.get(TARGET_CONFIG_NAME)

        if not config:
            print(f"--- ERROR: Конфигурация '{TARGET_CONFIG_NAME}' не найдена. ---")
            return

        run_single_backtest(data_df, config, persistence_config)

    elif RUN_MODE == "OPTIMIZE":
        base_config = STRATEGY_CONFIGS.get(TARGET_CONFIG_NAME)
        if not base_config:
            print(
                f"--- ERROR: Базовая конфигурация '{TARGET_CONFIG_NAME}' не найдена. ---"
            )
            return

        run_optimization(
            data_df,
            base_config,
            PARAMETER_SPACE,
            OPTIMIZATION_METRIC,
            persistence_config,
        )


if __name__ == "__main__":
    main()
