import time
import pandas as pd
import numpy as np
import itertools
import copy
from typing import Tuple, Dict, Any, List, Generator
from numba import njit  # НОВЫЙ ИМПОРТ ДЛЯ NUMBA

# Импорт модулей
from backtest.core.config import StrategyConfig, PersistenceConfig, FILE_PATH
from backtest.core.data_loader import load_data
from backtest.core.indicators import calculate_strategy_indicators
from backtest.core.analysis import calculate_metrics, display_results_rich, plot_results
from backtest.core.persistence import (
    persist_results,
    persist_optimization_result,
    StrategyConfig,
    PersistenceConfig,
    SCRIPT_DIR,
)


from rich.console import Console
import matplotlib.pyplot as plt

from backtest.core.data_loader import load_data
from backtest.core.config import FILE_PATH

# Инициализация консоли Rich
console = Console()


# def plot_results(
#     equity_for_plot: pd.Series,
#     drawdown_for_plot: pd.Series,
#     trades_df: pd.DataFrame,
#     initial_capital: float,
#     metric_name: str,
#     best_metrics: Dict[str, Any],
# ):
#     """Строит график кривой эквити и просадки."""
#     try:
#         if equity_for_plot.empty:
#             console.print("[PLOT] Нет данных для построения графика.")
#             return

#         # Индекс для графика (количество сделок)
#         trade_indices = np.arange(len(equity_for_plot))

#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#         fig.suptitle(f"Backtest Results (Optimized for {metric_name})", fontsize=16)

#         # График Эквити
#         ax1.plot(
#             trade_indices, equity_for_plot.values, label="Equity Curve", color="green"
#         )
#         ax1.axhline(
#             initial_capital,
#             color="grey",
#             linestyle="--",
#             alpha=0.7,
#             label="Initial Capital",
#         )
#         ax1.set_title("Кривая Эквити (Equity Curve)", fontsize=14)
#         ax1.set_ylabel("Капитал (USD)", fontsize=12)
#         ax1.grid(True, linestyle="--", alpha=0.6)
#         ax1.legend()

#         # График Просадки
#         # drawdown_for_plot уже нормализован и включает 0.0 в начале
#         ax2.fill_between(
#             trade_indices, 0, drawdown_for_plot.values * 100, color="red", alpha=0.4
#         )
#         ax2.plot(
#             trade_indices,
#             drawdown_for_plot.values * 100,
#             color="red",
#             label="Drawdown (%)",
#         )
#         ax2.set_title(
#             f"Относительная Просадка (Max DD: {best_metrics['Max Drawdown (%)']:.2f}%)",
#             fontsize=14,
#         )
#         ax2.set_ylabel("Просадка (%)", fontsize=12)
#         ax2.set_xlabel("Номер закрытой сделки", fontsize=12)
#         ax2.set_ylim(bottom=0)
#         ax2.grid(True, linestyle="--", alpha=0.6)
#         ax2.legend()

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Учитываем заголовок
#         plt.show()

#     except ImportError:
#         console.print(
#             "[bold red][PLOT ERROR][/bold red] Не удалось построить график. Установите matplotlib: [yellow]pip install matplotlib[/yellow]"
#         )
#     except Exception as e:
#         console.print(
#             f"[bold red][PLOT ERROR][/bold red] Произошла ошибка при построении графика: {e}"
#         )


# =========================================================================
# === ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ПЕРЕКЛЮЧАТЕЛИ РЕЖИМОВ ===
# =========================================================================

# --- ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМА ---
# Установите:
# - "SINGLE" для запуска одной стратегии (см. TARGET_CONFIG_NAME)
# - "OPTIMIZE" для запуска цикла оптимизации (см. PARAMETER_SPACE)
RUN_MODE = "OPTIMIZE"

# Какую стратегию запускать в режиме "SINGLE"
TARGET_CONFIG_NAME = "EMA_RSI_ATR_Strategy"

# =========================================================================
# === КОНФИГУРАЦИЯ СТРАТЕГИЙ И ПАРАМЕТРОВ ОПТИМИЗАЦИИ ===
# =========================================================================

# --- 1. ПАРАМЕТРИЧЕСКОЕ ПРОСТРАНСТВО ДЛЯ ОПТИМИЗАЦИИ ---
# Перебираемые параметры должны быть включены в StrategyConfig.indicator_set
PARAMETER_SPACE = {
    "EMA_TREND": {
        "fast_len": [5, 9, 13],  # Перебираем длины быстрой EMA
        "slow_len": [21, 34, 50],  # Перебираем длины медленной EMA
    },
    "RSI": {
        "rsi_len": [14, 20],  # Перебираем длины RSI
    },
    # Для целей оптимизации также меняем параметры риска/дохода
    "EXIT": {
        "target_roi_percent": [0.5, 1.0],
        "risk_roi_percent": [0.5, 0.8],
    },
}


# --- 2. КОНФИГУРАЦИЯ СТРАТЕГИЙ ---
# Здесь определяются все базовые настройки стратегий
STRATEGY_CONFIGS: Dict[str, StrategyConfig] = {
    "EMA_RSI_ATR_Strategy": StrategyConfig(
        initial_capital=10000.0,
        leverage=10.0,
        target_roi_percent=0.8,  # Процент дохода от входа (0.8%)
        risk_roi_percent=0.5,  # Процент риска от входа (0.5%)
        # Индикаторы, которые будут рассчитаны (и их параметры по умолчанию)
        indicator_set={
            "EMA_TREND": {"fast_len": 9, "slow_len": 21},
            "RSI": {"rsi_len": 14, "overbought": 70, "oversold": 30},
            "ATR_EXIT": {"atr_len": 14, "atr_multiplier": 1.5},
        },
    ),
}

# =========================================================================
# === МОДУЛЬ 4: NUMBA ЯДРО БЭКТЕСТА (@njit) ===
# =========================================================================


@njit
def _numba_backtest_core(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    rsi: np.ndarray,
    # atr: np.ndarray, # ATR пока не используется в этой упрощенной логике
    leverage: float,
    target_roi_perc: float,
    risk_roi_perc: float,
) -> Tuple[np.ndarray, float]:
    """
    Чистое ядро бэктеста, оптимизированное с помощью Numba (@njit).

    Расчеты:
    - Вход: Пересечение EMA + Фильтр RSI (импульсный контртренд).
    - Выход: Фиксированные Take Profit/Stop Loss.

    Возвращает 2D массив сделок и конечное значение эквити (в долях от начального капитала).
    """

    num_bars = len(close_prices)
    # Используем динамический список для сбора сделок (оптимизация для Numba)
    trades = []

    # Состояние позиций
    is_in_position = False
    side = 0  # 1: Long, -1: Short, 0: None
    entry_index = -1
    entry_price = 0.0

    # В Numba ядре работаем с прибылью/убытком, а не с эквити (начинаем с 1.0)
    current_equity = 1.0

    # Преобразуем проценты в доли (0.01 = 1%)
    target_roi = target_roi_perc / 100.0
    risk_roi = risk_roi_perc / 100.0

    # Параметры для сделки
    tp_price = 0.0
    sl_price = 0.0
    position_size_perc = 0.10  # 10% капитала на сделку

    # ----------------------------------------------------------------------
    # --- Основной цикл бэктеста ---
    # ----------------------------------------------------------------------
    # Начинаем с 1, так как смотрим на индикаторы предыдущей свечи (i-1)
    for i in range(1, num_bars):
        current_close = close_prices[i]

        # 1. ОБРАБОТКА ОТКРЫТОЙ ПОЗИЦИИ (Exit Logic)
        is_exit = False
        if is_in_position:
            pnl_perc = 0.0
            liquidation = 0  # В этой модели ликвидация не реализована

            # Проверка выхода Long (TP/SL)
            if side == 1:
                # Проверка TP (цена High достигла TP)
                if high_prices[i] >= tp_price:
                    exit_price = tp_price
                    # PNL %: ((Exit / Entry) - 1) * Leverage
                    pnl_perc = ((exit_price / entry_price) - 1.0) * leverage
                    is_exit = True
                # Проверка SL (цена Low достигла SL)
                elif low_prices[i] <= sl_price:
                    exit_price = sl_price
                    pnl_perc = ((exit_price / entry_price) - 1.0) * leverage
                    is_exit = True

            # Проверка выхода Short (TP/SL)
            elif side == -1:
                # Проверка TP (цена Low достигла TP)
                if low_prices[i] <= tp_price:
                    exit_price = tp_price
                    # PNL %: (1 - (Exit / Entry)) * Leverage
                    pnl_perc = (1.0 - (exit_price / entry_price)) * leverage
                    is_exit = True
                # Проверка SL (цена High достигла SL)
                elif high_prices[i] >= sl_price:
                    exit_price = sl_price
                    pnl_perc = (1.0 - (exit_price / entry_price)) * leverage
                    is_exit = True

            if is_exit:
                # Обновляем эквити в долях
                current_equity *= 1.0 + pnl_perc

                # Объем позиции (в USD, в долях от Initial Capital, для логгирования)
                volume_usd_ratio = position_size_perc * leverage

                # Запись сделки: [Закрытие Индекс, Индекс Входа, Сторона, Цена Входа, Цена Выхода, PNL %, Объем USD Ratio, Ликвидация]
                trade_record = np.array(
                    [
                        float(i),
                        float(entry_index),
                        float(side),
                        entry_price,
                        exit_price,
                        pnl_perc,
                        volume_usd_ratio,
                        float(liquidation),
                    ]
                )
                trades.append(trade_record)

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
                entry_price = current_close  # Вход по цене закрытия текущей свечи

                # Расчет TP/SL: Entry +/- (Entry * ROI_Perc)
                tp_price = entry_price * (1.0 + target_roi)
                sl_price = entry_price * (1.0 - risk_roi)

            # Вход Short
            elif short_signal:
                is_in_position = True
                side = -1
                entry_index = i
                entry_price = current_close

                # Расчет TP/SL: Entry +/- (Entry * ROI_Perc)
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

        trade_record = np.array(
            [
                float(num_bars - 1),
                float(entry_index),
                float(side),
                entry_price,
                exit_price,
                pnl_perc,
                volume_usd_ratio,
                0.0,
            ]
        )
        trades.append(trade_record)

    # Преобразование списка массивов в один 2D массив NumPy
    if len(trades) > 0:
        trades_array = np.array(trades)
    else:
        # Важно вернуть пустой массив правильной размерности
        trades_array = np.empty((0, 8))

    return trades_array, current_equity


# =========================================================================
# === МОДУЛЬ 5: ДВИГАТЕЛЬ БЭКТЕСТА (Python Wrapper) ===
# =========================================================================


def backtest_engine(
    df: pd.DataFrame, config: StrategyConfig
) -> Tuple[pd.DataFrame, float]:
    """
    Основной двигатель бэктеста. Подготавливает данные и вызывает ядро Numba.

    :param df: DataFrame с OHLCV и рассчитанными индикаторами.
    :param config: Объект StrategyConfig.
    :return: (trades_df, final_equity)
    """

    # 1. Подготовка данных для Numba (только массивы NumPy)
    close_prices = df["close"].values.astype(np.float64)
    high_prices = df["high"].values.astype(np.float64)
    low_prices = df["low"].values.astype(np.float64)

    # Индикаторы, необходимые для Numba ядра
    ema_fast = df["ema_fast"].values.astype(np.float64)
    ema_slow = df["ema_slow"].values.astype(np.float64)
    rsi = df["rsi_val"].values.astype(np.float64)
    # atr = df["atr_val"].values.astype(np.float64) # Закомментирован, т.к. не используется в Numba ядре

    # 2. Вызов ядра Numba
    trades_array, final_equity_ratio = _numba_backtest_core(
        close_prices,
        high_prices,
        low_prices,
        ema_fast,
        ema_slow,
        rsi,
        config.leverage,
        config.target_roi_percent,  # Передаем проценты, а не доли
        config.risk_roi_percent,  # Передаем проценты, а не доли
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
            "volume_usd_ratio",  # Это доля от начального капитала * плечо
            "liquidation",
        ],
    )

    # Преобразование индексов обратно в целые числа и сопоставление с датами
    trades_df["exit_index"] = trades_df["exit_index"].astype(int)
    trades_df["entry_index"] = trades_df["entry_index"].astype(int)

    # Сопоставление индексов с временными метками
    trades_df["entry_time"] = df.loc[trades_df["entry_index"], "timestamp"].values
    trades_df["exit_time"] = df.loc[trades_df["exit_index"], "timestamp"].values

    # Расчет PNL в абсолютных единицах (в валюте)
    # PNL_абсолютный = PNL_perc * (Initial_Capital * Position_Size_Perc * Leverage)
    # Где Position_Size_Perc = 0.10, а Leverage и Initial_Capital берем из config
    # Volume_USD_Ratio = 0.10 * Leverage
    trade_volume_usd = config.initial_capital * trades_df["volume_usd_ratio"]

    # PNL в валюте
    trades_df["pnl"] = trades_df["pnl_perc"] * trade_volume_usd

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

    :param config: Конфигурация стратегии.
    :param data_df: Исходные данные OHLCV.
    :return: (trades_df, metrics, execution_time)
    """

    start_time = time.time()

    # 1. Расчет индикаторов (уже оптимизирован с помощью TALIB в indicators.py)
    df_with_indicators = calculate_strategy_indicators(data_df, config)

    # Проверка, что DF не пуст после очистки
    if df_with_indicators.empty:
        print("--- ERROR: DataFrame пуст после расчета индикаторов и удаления NaN. ---")
        return pd.DataFrame(), {}, time.time() - start_time

    # 2. Выполнение бэктеста (ОПТИМИЗИРОВАНО NUMBA)
    # Вызываем нашу новую функцию backtest_engine
    trades_df, final_equity = backtest_engine(df_with_indicators, config)

    # 3. Анализ результатов
    metrics, _, _ = calculate_metrics(trades_df, config.initial_capital, final_equity)

    execution_time = time.time() - start_time

    return trades_df, metrics, execution_time


# =========================================================================
# === ОСНОВНАЯ ФУНКЦИЯ (MAIN) ===
# =========================================================================


def main():
    """
    Основная функция для запуска бэктеста или оптимизации.
    """
    print("--- ЗАПУСК СИСТЕМЫ БЭКТЕСТА ---")

    # 1. Загрузка данных
    # Примечание: data_loader.py генерирует mock-данные
    data_df = load_data(str(FILE_PATH))

    if data_df.empty:
        print("--- ERROR: Не удалось загрузить данные. Выход. ---")
        return

    # --- КОНФИГУРАЦИЯ СОХРАНЕНИЯ (Общая) ---
    persistence_config = PersistenceConfig(
        save_to_sqlite=False,  # Измените на True для сохранения в SQLite
        save_to_csv=True,
        save_to_txt=False,
        table_name="backtest_trades",
    )

    if RUN_MODE == "SINGLE":
        print(f"\n--- РЕЖИМ: ОДИНОЧНЫЙ ПРОГОН ({TARGET_CONFIG_NAME}) ---")
        config = STRATEGY_CONFIGS.get(TARGET_CONFIG_NAME)

        if not config:
            print(f"--- ERROR: Конфигурация '{TARGET_CONFIG_NAME}' не найдена. ---")
            return

        trades_df, metrics, execution_time = run_backtest(config, data_df)

        print("\n--- РЕЗУЛЬТАТЫ ОДИНОЧНОГО ПРОГОНА ---")
        display_results_rich(metrics, trades_df, execution_time)

        if not trades_df.empty:
            # 4. Сохранение и График
            print("\n--- ГРАФИК И СОХРАНЕНИЕ ---")
            # Отображение графика
            plot_results(trades_df, config.initial_capital)
            # Сохранение результатов
            persist_results(trades_df, metrics, config, persistence_config)

    elif RUN_MODE == "OPTIMIZE":
        print("\n--- РЕЖИМ: ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ---")

        # Получаем базовую конфигурацию
        base_config = STRATEGY_CONFIGS.get(TARGET_CONFIG_NAME)
        if not base_config:
            print(
                f"--- ERROR: Базовая конфигурация '{TARGET_CONFIG_NAME}' не найдена. ---"
            )
            return

        # 1. Генерация всех комбинаций параметров
        param_keys = []
        param_values = []

        # Обход параметров индикаторов
        for ind_name, ind_params in PARAMETER_SPACE.items():
            if ind_name in base_config.indicator_set or ind_name == "EXIT":
                for param_key, values in ind_params.items():
                    param_keys.append((ind_name, param_key))
                    param_values.append(values)

        # Генератор для всех комбинаций
        param_combinations = itertools.product(*param_values)

        total_runs = np.prod([len(v) for v in param_values])
        if total_runs == 0:
            print("--- WARNING: Не найдено параметров для оптимизации. ---")
            return

        print(f"--- INFO: Всего комбинаций для оптимизации: {int(total_runs)} ---")

        best_metric_value = -np.inf  # Для максимизации 'Total Return (%)'
        best_config = None
        best_metrics = None

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

            # Проверка логичности параметров (например, fast_len < slow_len)
            ema_fast_len = current_config.indicator_set.get("EMA_TREND", {}).get(
                "fast_len", 0
            )
            ema_slow_len = current_config.indicator_set.get("EMA_TREND", {}).get(
                "slow_len", 0
            )

            if ema_fast_len >= ema_slow_len:
                # print(f"--- SKIP: Нелогичные EMA длины: Fast={ema_fast_len}, Slow={ema_slow_len}")
                continue

            # Запуск бэктеста
            _, metrics, _ = run_backtest(current_config, data_df)

            # 3. Оценка результата
            # Цель: Максимизировать Общую Доходность (Total Return)
            current_metric_value = metrics.get("Total Return (%)", -np.inf)

            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_config = current_config
                best_metrics = metrics

                # print(f"--- NEW BEST: Return={current_metric_value:.2f}% (Run {run_count}/{int(total_runs)})")

        optimization_end_time = time.time()
        execution_time_opt = optimization_end_time - optimization_start_time

        # 4. Вывод и сохранение лучших результатов
        if best_config and best_metrics:
            print("\n--- РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ (ЛУЧШИЙ ПРОГОН) ---")
            print(f"Общее время оптимизации: {execution_time_opt:.2f} сек.")
            print(f"Проверено прогонов: {run_count}")
            print("\nПараметры лучшей конфигурации:")
            # Выводим только оптимизированные параметры для примера
            print(
                f"EMA Fast: {best_config.indicator_set.get('EMA_TREND', {}).get('fast_len', 'N/A')}"
            )
            print(
                f"EMA Slow: {best_config.indicator_set.get('EMA_TREND', {}).get('slow_len', 'N/A')}"
            )
            print(
                f"RSI Len: {best_config.indicator_set.get('RSI', {}).get('rsi_len', 'N/A')}"
            )

            display_results_rich(best_metrics, pd.DataFrame(), execution_time_opt)

            # НОВОЕ: Сохраняем лучшую конфигурацию и ее метрики
            print("\n--- СОХРАНЕНИЕ ЛУЧШЕГО РЕЗУЛЬТАТА ---")
            if best_config and best_metrics:
                # Используем новую функцию для сохранения лучших параметров
                persist_optimization_result(
                    best_config, best_metrics, persistence_config
                )
            else:
                print(
                    "[PERSIST] Лучшая конфигурация или метрики не найдены (скорее всего, не было сделок ни в одном прогоне)."
                )

        else:
            print("\n--- ОШИБКА ОПТИМИЗАЦИИ ---")
            print(
                "Не удалось найти лучшую конфигурацию. Все прогоны, вероятно, не смогли совершить сделки или их 'Total Return (%)' был -Inf."
            )


if __name__ == "__main__":
    main()
