import time
import pandas as pd
import numpy as np
import itertools
import copy
from typing import Tuple, Dict, Any, List, Generator

# Импорт модулей
# ПРИМЕЧАНИЕ: В реальном проекте StrategyConfig, PersistenceConfig, FILE_PATH
# импортируются из backtest.core.config
# Здесь используются заглушки, определенные в persistence.py для компиляции
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


# =========================================================================
# === ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ПЕРЕКЛЮЧАТЕЛИ РЕЖИМОВ ===
# =========================================================================

# --- ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМА ---
# Установите:
# - "SINGLE" для запуска одной стратегии (см. TARGET_CONFIG_NAME)
# - "OPTIMIZE" для запуска цикла оптимизации (см. PARAMETER_SPACE)
RUN_MODE = "OPTIMIZE"

# Какую стратегию запускать в режиме "SINGLE"
TARGET_CONFIG_NAME = "HFT_SCALPER"

# Пространство параметров для перебора (Grid Search)
PARAMETER_SPACE = {
    # Параметры Риск/Прибыль
    "target_roi_percent": [1.5, 2.0],
    "risk_roi_percent": [0.8, 1.0],
    # Параметры EMA_TREND
    "EMA_TREND_fast_len": [10, 12],
    "EMA_TREND_slow_len": [24, 26],
    # Параметры RSI (пример)
    "RSI_rsi_len": [7, 10],
}

# --- ЗАГЛУШКА ДЛЯ CONFIG_MAP ---
# В реальном коде замените это на импорт из scalping_strategies.py
BASE_STRATEGY_CONFIG_TEMPLATE = StrategyConfig(
    initial_capital=100.0,
    leverage=20.0,
    target_roi_percent=2.0,
    risk_roi_percent=1.0,
    indicator_set={
        "EMA_TREND": {"fast_len": 12, "slow_len": 26},
        "ATR_EXIT": {"atr_len": 20, "min_multiplier": 2.0},
        "RSI": {"rsi_len": 7},
        # ... другие индикаторы
    },
)
CONFIG_MAP = {
    "HFT_SCALPER": BASE_STRATEGY_CONFIG_TEMPLATE,
    "BREAKOUT_SCALPER": BASE_STRATEGY_CONFIG_TEMPLATE,
    "MR_SCALPER": BASE_STRATEGY_CONFIG_TEMPLATE,
}
# --- КОНЕЦ ЗАГЛУШКИ ---


# =========================================================================
# === МОДУЛЬ 3: ЯДРО БЭКТЕСТЕРА (NUMBA SIMULATION MOCK) ===
# =========================================================================


# @njit # Оставляем закомментированным, так как njit требует чистых numba-типов
def run_backtest(
    df_processed: pd.DataFrame, config: StrategyConfig
) -> Tuple[pd.DataFrame, float]:
    """
    ЗАГЛУШКА ЯДРА БЭКТЕСТЕРА (Numba-ядро).
    Она имитирует логику открытия/закрытия позиций на основе индикаторов
    и возвращает DataFrame с результатами сделок.
    """

    # 1. Подготовка массивов для Numba (имитация)
    close_prices = df_processed["close"].values
    fast_ema = df_processed.get(
        "ema_fast", pd.Series(np.zeros(len(df_processed)))
    ).values
    slow_ema = df_processed.get(
        "ema_slow", pd.Series(np.zeros(len(df_processed)))
    ).values
    atr_val = df_processed.get("atr_val", pd.Series(np.zeros(len(df_processed)))).values

    # 2. Параметры
    initial_capital = config.initial_capital
    leverage = config.leverage

    current_equity = initial_capital
    trades = []

    # Имитация входа в позицию (Long, если EMA Fast > EMA Slow)
    in_position = False

    # Используем уникальный seed для каждого прогона, чтобы результаты оптимизации
    # менялись (в данном случае это просто демонстрация)
    np.random.seed(int(time.time() * 1000) % 1000)

    for i in range(len(df_processed)):
        current_price = close_prices[i]

        # Имитация сигнала на вход (Простое пересечение EMA)
        long_condition = (
            fast_ema[i] > slow_ema[i]
            and fast_ema[i - 1] <= slow_ema[i - 1]
            and not in_position
        )

        if (
            long_condition and i > 1 and np.random.rand() < 0.2
        ):  # Только 20% сигналов реализуются
            entry_price = current_price

            # Расчет Take Profit и Stop Loss на основе ATR (для демонстрации)
            atr_multiplier = config.indicator_set["ATR_EXIT"].get("min_multiplier", 2.0)
            tp_abs = atr_val[i] * atr_multiplier * 0.1
            sl_abs = atr_val[i] * (atr_multiplier / 2) * 0.1

            take_profit = entry_price + tp_abs
            stop_loss = entry_price - sl_abs

            # Эффективный размер позиции (с учетом leverage)
            position_size_usd = current_equity * leverage

            in_position = True

            trade_data = {
                "side": "LONG",
                "entry_time": df_processed["timestamp"].iloc[i],
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "position_size": position_size_usd,
            }

        # Имитация закрытия позиции (Take Profit или Stop Loss)
        if in_position:
            if i + 1 < len(df_processed):
                next_close = close_prices[i + 1]

                # Имитация PNL на основе параметров ROI и leverage
                pnl_ratio_mock = (config.target_roi_percent / 100) * (
                    np.random.uniform(0.8, 1.2)
                )

                # Допустим, 60% сделок выигрышные
                if np.random.rand() > 0.4:
                    pnl = pnl_ratio_mock * initial_capital * leverage
                    close_reason = "Take Profit"
                else:
                    pnl = (
                        -(config.risk_roi_percent / 100)
                        * initial_capital
                        * leverage
                        * np.random.uniform(0.9, 1.1)
                    )
                    close_reason = "Stop Loss"

                # Имитация завершения сделки каждые 2-10 баров
                if np.random.rand() < 0.3:
                    current_equity += pnl
                    in_position = False

                    trades.append(
                        {
                            "entry_time": trade_data["entry_time"],
                            "side": trade_data["side"],
                            "entry_price": trade_data["entry_price"],
                            "exit_time": df_processed["timestamp"].iloc[i + 1],
                            "exit_price": next_close,
                            "pnl": pnl,
                            "close_reason": close_reason,
                            "liquidation": 0,
                            "equity_after_trade": current_equity,
                        }
                    )

    # Если остались незакрытые сделки в конце
    if in_position:
        pnl = (
            (np.random.rand() - 0.5) * initial_capital * leverage * 0.5
        )  # Случайный PnL для закрытия
        current_equity += pnl
        trades.append(
            {
                "entry_time": trade_data["entry_time"],
                "side": trade_data["side"],
                "entry_price": trade_data["entry_price"],
                "exit_time": df_processed["timestamp"].iloc[-1],
                "exit_price": close_prices[-1],
                "pnl": pnl,
                "close_reason": "End of Test",
                "liquidation": 0,
                "equity_after_trade": current_equity,
            }
        )

    trades_df = pd.DataFrame(trades)
    final_equity = current_equity

    if trades_df.empty:
        # Создание пустого DataFrame со всеми колонками
        trades_df = pd.DataFrame(
            columns=[
                "entry_time",
                "side",
                "entry_price",
                "exit_price",
                "pnl",
                "close_reason",
                "liquidation",
                "equity_after_trade",
            ]
        )

    return trades_df, final_equity


# =========================================================================
# === ФУНКЦИИ ОПТИМИЗАЦИИ И ОРКЕСТРАЦИИ ===
# =========================================================================


def generate_combinations(
    params: Dict[str, List[Any]],
) -> Generator[Dict[str, Any], None, None]:
    """Генератор всех комбинаций параметров (Grid Search)."""
    keys = params.keys()
    values = params.values()
    # Создаем декартово произведение всех списков значений
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def run_optimization(
    initial_config: StrategyConfig,
    param_space: Dict[str, List[Any]],
    df_data: pd.DataFrame,
) -> Tuple[StrategyConfig, Dict[str, Any]]:
    """
    Прогоняет бэктест для всех комбинаций параметров и находит лучшую.
    Оценивается по метрике 'Total Return (%)'.
    """

    best_config = None
    best_metrics = None
    best_score = -np.inf  # Инициализируем минимальной возможной оценкой

    all_combinations = list(generate_combinations(param_space))
    total_runs = len(all_combinations)
    current_run = 0

    print(f"\n[OPTIMIZER] Начинаем перебор {total_runs} комбинаций...")

    for combo in all_combinations:
        current_run += 1
        start_time_run = time.time()

        # 1. Клонируем исходную конфигурацию
        temp_config = copy.deepcopy(initial_config)

        # 2. Обновляем параметры конфигурации в соответствии с текущей комбинацией
        temp_config.target_roi_percent = combo.get(
            "target_roi_percent", temp_config.target_roi_percent
        )
        temp_config.risk_roi_percent = combo.get(
            "risk_roi_percent", temp_config.risk_roi_percent
        )

        # Обновление параметров индикаторов
        if "EMA_TREND_fast_len" in combo:
            temp_config.indicator_set["EMA_TREND"]["fast_len"] = combo[
                "EMA_TREND_fast_len"
            ]
        if "EMA_TREND_slow_len" in combo:
            temp_config.indicator_set["EMA_TREND"]["slow_len"] = combo[
                "EMA_TREND_slow_len"
            ]
        if "RSI_rsi_len" in combo:
            temp_config.indicator_set["RSI"]["rsi_len"] = combo["RSI_rsi_len"]

        # 3. Расчет индикаторов и запуск бэктеста
        df_processed = calculate_strategy_indicators(df_data, temp_config)
        trades_df, final_equity = run_backtest(df_processed, temp_config)

        end_time_run = time.time()
        execution_time_run = end_time_run - start_time_run

        # 4. Расчет метрик
        metrics, _, _ = calculate_metrics(
            trades_df, temp_config.initial_capital, final_equity
        )
        metrics["Время выполнения (Numba)"] = (
            execution_time_run  # Сохраняем время прогона
        )

        # --- БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ КЛЮЧА ---
        current_score = metrics.get(
            "Total Return (%)", -np.inf
        )  # Используем -inf для надежности

        # 5. Сравнение и обновление лучшего результата
        # ИСПРАВЛЕНИЕ: Используем >=, чтобы best_config был установлен хотя бы один раз,
        # даже если все результаты равны -inf.
        if current_score >= best_score:
            best_score = current_score
            best_config = temp_config
            best_metrics = metrics

        print(
            f"[Run {current_run:0{len(str(total_runs))}}/{total_runs}] Return: {current_score:.2f}% | Config: {combo}"
        )

    print("\n" + "=" * 70)
    print(f"=== ОПТИМИЗАЦИЯ ЗАВЕРШЕНА. НАЙДЕН ЛУЧШИЙ РЕЗУЛЬТАТ: {best_score:.2f}% ===")
    print("=" * 70)

    return best_config, best_metrics


def run_single_backtest(
    strategy_config: StrategyConfig,
    df_raw: pd.DataFrame,
    persistence_config: PersistenceConfig,
):
    """Выполняет один прогон стратегии и выводит результаты."""
    print(f"--- ЗАПУСК ОДИНОЧНОГО ПРОГОНА: {TARGET_CONFIG_NAME} ---")

    # 2. РАСЧЕТ ИНДИКАТОРОВ
    df_processed = calculate_strategy_indicators(df_raw, strategy_config)

    start_time = time.time()

    # 3. ЗАПУСК БЭКТЕСТА
    trades_df, final_equity = run_backtest(df_processed, strategy_config)

    end_time = time.time()
    execution_time = end_time - start_time

    # 4. АНАЛИЗ И ОТЧЕТНОСТЬ
    metrics, drawdown_for_plot, equity_curve = calculate_metrics(
        trades_df, strategy_config.initial_capital, final_equity
    )

    display_results_rich(metrics, trades_df, execution_time)

    # 5. ПЕРСИСТЕНТНОСТЬ
    persist_results(trades_df, persistence_config)

    # 6. ГРАФИКИ
    plot_results(
        trades_df, equity_curve, drawdown_for_plot, strategy_config.initial_capital
    )
    print(
        "\n[INFO] Для отображения графиков раскомментируйте строку 'plot_results(...)' в конце скрипта."
    )
    print("--- КОНЕЦ СИМУЛЯЦИИ ---")


# =========================================================================
# === ОСНОВНОЙ СКРИПТ: ОРКЕСТРАЦИЯ (ВЫБОР РЕЖИМА) ===
# =========================================================================

if __name__ == "__main__":

    print("--- ЗАПУСК БЭКТЕСТЕРА В РЕЖИМЕ ВЫБОРА ---")

    # Конфигурация сохранения
    persistence_config = PersistenceConfig(
        save_to_sqlite=False,
        save_to_csv=True,  # Включаем сохранение в CSV для демонстрации
        save_to_txt=False,
        table_name="scalper_test_run",
    )

    # 1. ЗАГРУЗКА ДАННЫХ (один раз)
    print(f"[INFO] Начинаем загрузку данных из {FILE_PATH}...")
    df_raw = load_data(str(FILE_PATH))

    # --- ЛОГИКА ВЫБОРА РЕЖИМА ---
    if RUN_MODE == "SINGLE":
        # 1.a. ОДИНОЧНЫЙ ПРОГОН
        if TARGET_CONFIG_NAME not in CONFIG_MAP:
            print(
                f"[ERROR] Конфигурация '{TARGET_CONFIG_NAME}' не найдена. Проверьте TARGET_CONFIG_NAME."
            )
        else:
            strategy_config = CONFIG_MAP[TARGET_CONFIG_NAME]
            run_single_backtest(strategy_config, df_raw, persistence_config)

    elif RUN_MODE == "OPTIMIZE":
        # 1.b. РЕЖИМ ОПТИМИЗАЦИИ
        # Берем базовую конфигурацию для параметров, которые не оптимизируются.
        base_strategy_config = CONFIG_MAP["HFT_SCALPER"]

        start_time_opt = time.time()
        best_config, best_metrics = run_optimization(
            base_strategy_config, PARAMETER_SPACE, df_raw
        )
        end_time_opt = time.time()
        execution_time_opt = end_time_opt - start_time_opt

        # --- ИСПРАВЛЕНИЕ 2: Проверка best_config на None ---
        if best_config:
            # Обновляем метрики лучшего прогона общим временем выполнения оптимизации
            if best_metrics:
                best_metrics["Время выполнения (Numba)"] = (
                    f"{execution_time_opt:.4f} сек."
                )

            # Вывод результатов оптимизации
            print("\n--- ЛУЧШАЯ НАЙДЕННАЯ КОНФИГУРАЦИЯ ---")
            print(f"Target ROI: {best_config.target_roi_percent}")
            print(f"Risk ROI: {best_config.risk_roi_percent}")
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
                "Не удалось найти лучшую конфигурацию. Все прогоны, вероятно, не смогли совершить сделки или их 'Total Return (%)' был равен -inf."
            )

    else:
        print(
            f"[ERROR] Неизвестный режим RUN_MODE: {RUN_MODE}. Используйте 'SINGLE' или 'OPTIMIZE'."
        )
