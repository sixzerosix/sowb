import time
import pandas as pd
import numpy as np
import itertools
import copy
from typing import Tuple, Dict, Any, List, Generator

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

# =========================================================================
# === ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ПЕРЕКЛЮЧАТЕЛИ РЕЖИМОВ ===
# =========================================================================

# --- ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМА ---
# Установите:
# - "SINGLE" для запуска одной стратегии (см. TARGET_CONFIG_NAME)
# - "OPTIMIZE" для запуска цикла оптимизации (см. PARAMETER_SPACE)
RUN_MODE = "OPTIMIZE"

# Какую стратегию запускать в режиме "SINGLE"
TARGET_CONFIG_NAME = "SIMPLE_EMA_RSI_ATR"


# --- ПРОСТРАНСТВО ПАРАМЕТРОВ ДЛЯ ОПТИМИЗАЦИИ ---
# Перебираемые параметры должны быть в виде списка.
PARAMETER_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "EMA_TREND": {"fast_len": [5, 7], "slow_len": [21, 34]},
    "RSI": {"rsi_len": [14, 10], "overbought": [70], "oversold": [30]},
    "ATR_EXIT": {"atr_len": [14], "atr_multiplier": [1.0, 1.5]},
}


# =========================================================================
# === МОДУЛЬ 4: ТОРГОВАЯ ЛОГИКА (ОСНОВНОЙ ЦИКЛ) ===
# =========================================================================


def run_backtest(
    data: pd.DataFrame, config: StrategyConfig, is_optimization: bool = False
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Основной цикл бэктеста: итерация по данным и исполнение сделок.

    :param data: DataFrame с данными и индикаторами.
    :param config: Объект StrategyConfig с параметрами.
    :param is_optimization: Флаг, указывающий, что это прогон оптимизации.
    :return: (Метрики, DataFrame сделок, DataFrame данных с сигналами)
    """
    df = data.copy()  # Рабочая копия данных
    current_capital = config.initial_capital
    position_size = 0.0  # Текущий размер позиции в USD
    is_long = False
    trades_list = []  # Список для записи результатов сделок
    # Множество для хранения индексов, где была открыта позиция (для предотвращения двойных входов)
    open_indices = set()

    # --- 1. РАСЧЕТ ИНДИКАТОРОВ ---
    df_with_indicators = calculate_strategy_indicators(df, config)

    # Проверка, что после удаления NaN осталось достаточно данных
    if len(df_with_indicators) < 100:
        print(
            "--- WARNING: Недостаточно данных после расчета индикаторов. Пропускаем прогон."
        )
        # >>> ИСПРАВЛЕНИЕ: Возвращаем пустые, но корректные результаты в случае пропуска
        return (
            {"Total Trades": 0, "Total PnL": 0.0},
            pd.DataFrame(),
            df_with_indicators,
        )

    # --- 2. ЦИКЛ СТРАТЕГИИ ---
    for i in range(len(df_with_indicators)):
        row = df_with_indicators.iloc[i]

        # --- СИГНАЛЫ (Пример: EMA Crossover + RSI Filter) ---
        ema_cross_up = row["ema_fast"] > row["ema_slow"]
        ema_cross_down = row["ema_fast"] < row["ema_slow"]
        is_oversold = row["rsi_val"] < config.indicator_set["RSI"]["oversold"]
        is_overbought = row["rsi_val"] > config.indicator_set["RSI"]["overbought"]

        # --- УПРАВЛЕНИЕ ПОЗИЦИЕЙ ---

        # 1. СИГНАЛ НА ОТКРЫТИЕ (LONG)
        if not is_long and ema_cross_up and is_oversold:
            # Расчет размера позиции (100% капитала * кредитное плечо)
            position_size = current_capital * config.leverage
            entry_price = row["close"]

            # Запись сделки (только открытие)
            trade = {
                "open_time": row["timestamp"],
                "side": "LONG",
                "entry_price": entry_price,
                "position_size": position_size,
                "exit_price": np.nan,
                "pnl": np.nan,
                "roi": np.nan,
                "duration": np.nan,
                "liquidation": 0,
            }
            is_long = True
            open_indices.add(i)

        # 2. СИГНАЛ НА ЗАКРЫТИЕ (EXIT)
        elif is_long:
            exit_signal = (
                ema_cross_down  # Трендовый разворот
                or is_overbought  # Фиксация прибыли
            )

            if exit_signal:
                exit_price = row["close"]
                pnl, roi, final_equity = calculate_trade_pnl(
                    trade["entry_price"],
                    exit_price,
                    trade["position_size"],
                    current_capital,
                    side="LONG",
                )

                # Завершение и запись сделки
                trade.update(
                    {
                        "exit_time": row["timestamp"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "roi": roi,
                        "duration": (
                            row["timestamp"] - trade["open_time"]
                        ).total_seconds()
                        / 60,  # в минутах
                        "equity_after_trade": final_equity,  # ОБЯЗАТЕЛЬНАЯ КОЛОНКА
                    }
                )

                trades_list.append(trade)
                current_capital = final_equity  # Обновляем капитал
                is_long = False
                position_size = 0.0

    # --- 3. ЗАКРЫТИЕ ОТКРЫТОЙ ПОЗИЦИИ (если цикл закончился) ---
    # Если позиция осталась открытой до конца данных, закрываем ее по последней цене
    if is_long and not df_with_indicators.empty:
        last_row = df_with_indicators.iloc[-1]
        exit_price = last_row["close"]
        pnl, roi, final_equity = calculate_trade_pnl(
            trade["entry_price"],
            exit_price,
            trade["position_size"],
            current_capital,
            side="LONG",
        )
        trade.update(
            {
                "exit_time": last_row["timestamp"],
                "exit_price": exit_price,
                "pnl": pnl,
                "roi": roi,
                "duration": (last_row["timestamp"] - trade["open_time"]).total_seconds()
                / 60,
                "equity_after_trade": final_equity,  # ОБЯЗАТЕЛЬНАЯ КОЛОНКА
            }
        )
        trades_list.append(trade)
        current_capital = final_equity

    # --- 4. ПОДГОТОВКА РЕЗУЛЬТАТОВ ---
    trades_df = pd.DataFrame(trades_list)
    final_equity = current_capital

    # >>> ИСПРАВЛЕНИЕ: Защита на случай, если сделок не было
    if trades_df.empty:
        print("--- WARNING: В данном прогоне сделки не совершались.")
        metrics = {
            "Total Trades": 0,
            "Total PnL": 0.0,
            "Final Equity": final_equity,
            "Return (%)": 0.0,
        }
        drawdown = pd.Series([0.0])
        equity_curve = pd.Series([final_equity])
        return metrics, pd.DataFrame(), df_with_indicators

    # --- 5. АНАЛИЗ И МЕТРИКИ ---
    metrics, drawdown, equity_curve = calculate_metrics(
        trades_df, config.initial_capital, final_equity
    )

    # В режиме оптимизации возвращаем только ключевые метрики для ранжирования
    if is_optimization:
        return metrics, trades_df, df_with_indicators

    return metrics, trades_df, df_with_indicators


def calculate_trade_pnl(
    entry_price: float,
    exit_price: float,
    position_size: float,
    current_capital: float,
    side: str,
) -> Tuple[float, float, float]:
    """Расчет PnL (в USD) и ROI (в %) для одной сделки."""

    # 1. Расчет базового PnL (изменение цены в %)
    if side == "LONG":
        pnl_percent = (exit_price - entry_price) / entry_price
    elif side == "SHORT":
        pnl_percent = (entry_price - exit_price) / exit_price
    else:
        raise ValueError("Неверное направление сделки (side).")

    # 2. Учет плеча (leverage) и расчет ROI от капитала
    # pnl_value = (position_size * pnl_percent)
    pnl_value = (
        current_capital * pnl_percent * (position_size / current_capital)
    )  # Упрощенный расчет PnL на основе размера позиции
    roi_percent = (pnl_value / current_capital) * 100

    # 3. Обновление капитала
    final_equity = current_capital + pnl_value

    return pnl_value, roi_percent, final_equity


# =========================================================================
# === МОДУЛЬ 5: ОПТИМИЗАЦИЯ ===
# =========================================================================


def generate_configs(
    parameter_space: Dict[str, Dict[str, List[Any]]], base_config: StrategyConfig
) -> Generator[StrategyConfig, None, None]:
    """Генератор, создающий комбинации StrategyConfig из пространства параметров."""

    # 1. Извлечение всех параметров и их значений
    all_params = []
    indicator_names = []

    for indicator_name, params in parameter_space.items():
        indicator_names.append(indicator_name)
        # Получаем список всех комбинаций для этого индикатора
        keys = list(params.keys())
        values = list(params.values())

        # itertools.product создает все комбинации значений
        param_combinations = list(itertools.product(*values))

        # Форматируем каждую комбинацию в виде словаря
        indicator_param_sets = [dict(zip(keys, combo)) for combo in param_combinations]
        all_params.append(indicator_param_sets)

    # 2. Создание всех комбинаций индикаторов
    # product для комбинаций словарей индикаторов
    for combo_of_indicator_sets in itertools.product(*all_params):
        new_config = copy.deepcopy(base_config)

        # Сборка итогового indicator_set
        final_indicator_set = {}
        for i, indicator_set in enumerate(combo_of_indicator_sets):
            final_indicator_set[indicator_names[i]] = indicator_set

        new_config.indicator_set = final_indicator_set
        yield new_config


def run_optimization(
    data: pd.DataFrame,
    parameter_space: Dict[str, Dict[str, List[Any]]],
    base_config: StrategyConfig,
    persistence_config: PersistenceConfig,
) -> Tuple[StrategyConfig | None, Dict[str, Any] | None, float]:
    """Запускает цикл оптимизации параметров стратегии."""

    start_time = time.time()
    best_total_pnl = -np.inf
    best_config = None
    best_metrics = None

    print("\n--- НАЧАЛО ОПТИМИЗАЦИИ ПАРАМЕТРОВ ---")

    # Используем генератор для перебора конфигураций
    for run_num, current_config in enumerate(
        generate_configs(parameter_space, base_config), 1
    ):
        print(f"\n[RUN {run_num}] Тестирование: {current_config.indicator_set}")

        # Запуск бэктеста с текущей конфигурацией
        metrics, trades_df, _ = run_backtest(data, current_config, is_optimization=True)

        # Критерий выбора лучшей стратегии: Total PnL (можно изменить на 'Return (%)' или 'Profit Factor')
        current_pnl = metrics.get("Total PnL", -np.inf)

        # Проверка и обновление лучшей конфигурации
        if current_pnl > best_total_pnl:
            best_total_pnl = current_pnl
            best_config = current_config
            best_metrics = metrics

            print(
                f"[RUN {run_num}] НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ: PnL = ${best_total_pnl:,.2f}"
            )

    execution_time = time.time() - start_time

    return best_config, best_metrics, execution_time


# =========================================================================
# === МОДУЛЬ 7: ОСНОВНАЯ ТОЧКА ВХОДА ===
# =========================================================================


def main():
    # Инициализация базовых конфигураций
    strategy_config = StrategyConfig(
        initial_capital=1000.0,
        leverage=1.0,
        target_roi_percent=1.0,
        risk_roi_percent=0.5,
        # Базовый набор индикаторов для SINGLE-прогона
        indicator_set={
            "EMA_TREND": {"fast_len": 9, "slow_len": 21},
            "RSI": {"rsi_len": 14, "overbought": 70, "oversold": 30},
            "ATR_EXIT": {"atr_len": 14, "atr_multiplier": 1.0},
        },
    )

    persistence_config = PersistenceConfig(
        save_to_sqlite=False,
        save_to_csv=False,
        save_to_txt=False,
        table_name="backtest_trades",
        optimization_table_name="optimization_results",
    )

    # --- 1. ЗАГРУЗКА ДАННЫХ ---
    # FILE_PATH берется из config.py
    data = load_data(FILE_PATH)
    print(f"[MAIN] Загружено {len(data)} свечей.")

    if data.empty:
        print("[MAIN] ОШИБКА: Не удалось загрузить или сгенерировать данные. Выход.")
        return

    # --- 2. ЗАПУСК ВЫБРАННОГО РЕЖИМА ---
    if RUN_MODE == "SINGLE":
        print(
            f"\n--- ЗАПУСК В РЕЖИМЕ '{RUN_MODE}' (Стратегия: {TARGET_CONFIG_NAME}) ---"
        )

        start_time = time.time()
        metrics, trades_df, df_with_signals = run_backtest(data, strategy_config)
        execution_time = time.time() - start_time

        print("\n--- РЕЗУЛЬТАТЫ ОДИНОЧНОГО ПРОГОНА ---")
        display_results_rich(metrics, trades_df, execution_time)

        if not trades_df.empty:
            # Сохранение и графики только для одиночного прогона
            persist_results(trades_df, persistence_config)

            # Для построения графиков используем данные, возвращенные из calculate_metrics
            _, drawdown, equity_curve = calculate_metrics(
                trades_df, strategy_config.initial_capital, metrics["Final Equity"]
            )
            plot_results(
                trades_df, strategy_config.initial_capital, equity_curve, drawdown
            )

    elif RUN_MODE == "OPTIMIZE":
        print(f"\n--- ЗАПУСК В РЕЖИМЕ '{RUN_MODE}' ---")

        best_config, best_metrics, execution_time_opt = run_optimization(
            data, PARAMETER_SPACE, strategy_config, persistence_config
        )

        if best_config and best_metrics:
            print("\n--- ЛУЧШАЯ КОНФИГУРАЦИЯ ---")
            print(f"PnL: ${best_metrics['Total PnL']:,.2f}")
            # Отображаем только оптимизированные параметры для примера
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
                "Не удалось найти лучшую конфигурацию. Все прогоны, вероятно, не смогли совершить сделки или их 'Total PnL' был слишком низким."
            )

    else:
        print(
            f"ОШИБКА: Неизвестный режим запуска: {RUN_MODE}. Используйте 'SINGLE' или 'OPTIMIZE'."
        )


if __name__ == "__main__":
    main()
