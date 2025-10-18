import time
import pandas as pd
import numpy as np
import itertools
import copy
from typing import Tuple, Dict, Any, List, Generator
from numba import jit

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

from backtest.core.strategy_configs import (
    STRATEGY_CONFIG_MR_SCALPER,
    STRATEGY_CONFIG_BREAKOUT_SCALPER,
    STRATEGY_CONFIG_HFT_SCALPER,
)

# =========================================================================
# === ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ПЕРЕКЛЮЧАТЕЛИ РЕЖИМОВ ===
# =========================================================================

# --- ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМА ---
# Установите:
# - "SINGLE" для запуска одной стратегии
# - "OPTIMIZE" для запуска цикла оптимизации
RUN_MODE = "OPTIMIZE"

# Какую стратегию запускать в режиме "SINGLE" (используйте вашу конфигурацию из config.py)
TARGET_CONFIG_NAME = "MR_SCALPER"


# --- ПРОСТРАНСТВО ПАРАМЕТРОВ ДЛЯ ОПТИМИЗАЦИИ ---
# Перебираемые параметры должны быть в виде списка.
PARAMETER_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "EMA_TREND": {"fast_len": [5, 7], "slow_len": [21, 34]},
    "RSI": {"rsi_len": [10, 14], "oversold": [30]},
    "ATR_EXIT": {"atr_len": [10, 14]},  # Добавляем ATR в оптимизацию
    "HTF_FILTER": {"period": ["30min", "1H"]},  # Оптимизация периода HTF
}


# =========================================================================
# === МОДУЛЬ 4: ТОРГОВАЯ ЛОГИКА (ОСНОВНОЙ ЦИКЛ) ===
# =========================================================================


# @jit(nopython=True) - Эта директива компилирует функцию в машинный код (C)
@jit(nopython=True)
def numba_backtest_core(
    closes: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    rsi_val: np.ndarray,
    atr_val: np.ndarray,  # ATR для динамического SL/TP
    macd_hist: np.ndarray,  # MACD для дополнительного фильтра
    oversold: float,
    overbought: float,
    target_roi_percent: float,
    risk_roi_percent: float,
    leverage: float,
    htf_trend_up: np.ndarray,  # HTF фильтр
) -> Tuple[np.ndarray, float]:
    """
    Высокоскоростной торговый цикл с полной логикой, скомпилированной с Numba.

    Вся логика входа, выхода, SL/TP и расчета PnL находится здесь.
    """

    current_capital = 1000.0  # Начальный капитал фиксирован для Numba
    is_long = False

    # Переменные для отслеживания текущей открытой позиции
    entry_price = 0.0
    open_idx = -1

    # Динамические SL/TP (в цене)
    stop_loss_price = 0.0
    take_profit_price = 0.0

    # Список для временного хранения данных сделок
    max_trades = len(closes) // 2
    trades_array = np.zeros((max_trades, 6), dtype=np.float64)
    trade_count = 0

    # Проходим по всем данным
    for i in range(len(closes)):
        current_close = closes[i]

        # --- СИГНАЛЫ И ФИЛЬТРЫ ---
        ema_cross_up = ema_fast[i] > ema_slow[i]
        ema_cross_down = ema_fast[i] < ema_slow[i]

        is_oversold_signal = rsi_val[i] < oversold
        is_overbought_signal = rsi_val[i] > overbought

        # MACD фильтр: гистограмма растет
        macd_filter_pass = macd_hist[i] > 0.0

        # HTF фильтр: Тренд на старшем ТФ должен быть восходящим
        htf_filter_pass = htf_trend_up[i]

        # --- УПРАВЛЕНИЕ ПОЗИЦИЕЙ ---

        # 1. ЗАКРЫТИЕ ПОЗИЦИИ (SL/TP)
        if is_long:
            pnl_value = 0.0
            exit_price = 0.0
            exit_reason = 0  # 0=Нет, 1=TP, 2=SL, 3=Trend, 4=End of data

            # Check Take Profit
            if current_close >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = 1

            # Check Stop Loss
            elif current_close <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = 2

            # Check Trend Reversal (EMA Cross Down OR Overbought RSI)
            # Добавим условие: если HTF тренд стал нисходящим, тоже закрываем
            elif ema_cross_down or is_overbought_signal or not htf_filter_pass:
                exit_price = current_close
                exit_reason = 3

            if exit_reason != 0:
                # Расчет PnL
                pnl_percent = (exit_price - entry_price) / entry_price
                pnl_value = current_capital * pnl_percent * leverage
                final_equity = current_capital + pnl_value

                # Запись сделки
                if trade_count < max_trades:
                    trades_array[trade_count] = np.array(
                        [
                            float(open_idx),
                            float(i),
                            entry_price,
                            exit_price,
                            pnl_value,
                            final_equity,
                        ]
                    )
                    trade_count += 1

                current_capital = final_equity  # Обновляем капитал
                is_long = False
                open_idx = -1
                stop_loss_price = 0.0
                take_profit_price = 0.0

        # 2. СИГНАЛ НА ОТКРЫТИЕ (LONG)
        if not is_long:

            # Полный сигнал входа LONG
            # (EMA-кросс UP) AND (RSI - перепроданность) AND (HTF - UP) AND (MACD - UP)
            entry_signal = (
                ema_cross_up
                and is_oversold_signal
                and htf_filter_pass
                and macd_filter_pass
            )

            if entry_signal:
                if trade_count < max_trades:
                    entry_price = current_close
                    open_idx = i
                    is_long = True

                    # Расчет SL и TP в цене на основе ATR и % риска/цели
                    atr_distance = atr_val[i]  # Берем ATR на текущей свече

                    # target_roi_percent и risk_roi_percent теперь множители ATR
                    # (хотя могут быть и множителями цены, для простоты берем ATR)
                    target_move_in_price = atr_distance * target_roi_percent
                    risk_move_in_price = atr_distance * risk_roi_percent

                    take_profit_price = entry_price + target_move_in_price
                    stop_loss_price = entry_price - risk_move_in_price

    # --- 3. ЗАКРЫТИЕ ОТКРЫТОЙ ПОЗИЦИИ (если цикл закончился) ---
    if is_long and open_idx != -1 and trade_count < max_trades:
        last_idx = len(closes) - 1
        exit_price = closes[last_idx]

        # Расчет PnL
        pnl_percent = (exit_price - entry_price) / entry_price
        pnl_value = current_capital * pnl_percent * leverage
        final_equity = current_capital + pnl_value

        # Запись последней сделки
        trades_array[trade_count] = np.array(
            [
                float(open_idx),
                float(last_idx),
                entry_price,
                exit_price,
                pnl_value,
                final_equity,
            ]
        )
        current_capital = final_equity
        trade_count += 1

    # Возвращаем только заполненную часть массива сделок и финальный капитал
    return trades_array[:trade_count], current_capital


def run_backtest(
    data: pd.DataFrame, config: StrategyConfig, is_optimization: bool = False
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Основной цикл бэктеста: итерация по данным и исполнение сделок (теперь с Numba).
    """
    df = data.copy()  # Рабочая копия данных
    initial_capital = config.initial_capital

    # --- 1. РАСЧЕТ ИНДИКАТОРОВ (ВЫПОЛНЯЕТСЯ В PYTHON) ---
    df_with_indicators = calculate_strategy_indicators(df, config)

    if len(df_with_indicators) < 100:
        print(
            "--- WARNING: Недостаточно данных после расчета индикаторов. Пропускаем прогон."
        )
        return (
            {
                "Total Trades": 0,
                "Total PnL": 0.0,
                "Final Equity": initial_capital,
                "Return (%)": 0.0,
            },
            pd.DataFrame(),
            df_with_indicators,
        )

    # --- 2. ПОДГОТОВКА ДАННЫХ ДЛЯ NUMBA (ТОЛЬКО NUMPY МАССИВЫ) ---
    # Извлекаем все необходимые данные для Numba-ядра.
    closes = df_with_indicators["close"].values

    # EMA/RSI для сигналов
    ema_fast = df_with_indicators.get("ema_fast", np.array([])).values
    ema_slow = df_with_indicators.get("ema_slow", np.array([])).values
    rsi_val = df_with_indicators.get("rsi_val", np.array([])).values

    # ATR для динамического SL/TP
    # Используем значение по умолчанию 0.01, если ATR не рассчитан
    atr_val = df_with_indicators.get("atr_val", np.full(len(closes), 0.01)).values

    # MACD для фильтра
    macd_hist = df_with_indicators.get("macd_hist", np.full(len(closes), 1.0)).values

    # HTF ФИЛЬТР (Используем htf_trend_up)
    if "htf_trend_up" in df_with_indicators.columns:
        # Логика фильтра: HTF тренд UP, если быстрая EMA > медленной EMA
        htf_trend_up_array = df_with_indicators["htf_trend_up"].values
    else:
        # Если HTF-колонки отсутствуют, фильтр HTF отключен (по умолчанию True).
        htf_trend_up_array = np.full(len(closes), True, dtype=np.bool_)

    # Параметры из Config
    oversold = config.indicator_set.get("RSI", {}).get("oversold", 30.0)
    overbought = config.indicator_set.get("RSI", {}).get("overbought", 70.0)

    target_roi_percent = config.target_roi_percent
    risk_roi_percent = config.risk_roi_percent

    # --- 3. ВЫЗОВ NUMBA-КОМПИЛИРОВАННОГО ЯДРА ---
    # ПРИМЕЧАНИЕ: Если вы захотите использовать SWING, FIBO, BBANDS и т.д.,
    # вам нужно будет добавить их массивы здесь и в def numba_backtest_core.

    trades_array, final_equity = numba_backtest_core(
        closes,
        ema_fast,
        ema_slow,
        rsi_val,
        atr_val,
        macd_hist,
        oversold,
        overbought,
        target_roi_percent,
        risk_roi_percent,
        config.leverage,
        htf_trend_up_array,
    )

    # (Остальная часть run_backtest остается прежней)

    # --- 4. ПРЕОБРАЗОВАНИЕ РЕЗУЛЬТАТОВ ОБРАТНО В DATAFRAME ---
    trades_df = pd.DataFrame(
        trades_array,
        columns=[
            "open_idx",
            "close_idx",
            "entry_price",
            "exit_price",
            "pnl",
            "equity_after_trade",
        ],
    )

    if trades_df.empty:
        print("--- WARNING: В данном прогоне сделки не совершались.")
        metrics = {
            "Total Trades": 0,
            "Total PnL": 0.0,
            "Final Equity": final_equity,
            "Return (%)": 0.0,
        }
        return metrics, pd.DataFrame(), df_with_indicators

    # Добавляем временные метки (нужно взять из исходного DF по индексам)
    trades_df["open_time"] = df_with_indicators.iloc[
        trades_df["open_idx"].astype(int).values
    ]["timestamp"].values
    trades_df["exit_time"] = df_with_indicators.iloc[
        trades_df["close_idx"].astype(int).values
    ]["timestamp"].values

    # Расчет ROI и длительности
    trades_df["roi"] = (trades_df["pnl"] / initial_capital) * 100
    trades_df["duration"] = (
        trades_df["exit_time"] - trades_df["open_time"]
    ).dt.total_seconds() / 60  # в минутах
    trades_df["side"] = "LONG"  # Статическая информация для данной стратегии

    # --- 5. АНАЛИЗ И МЕТРИКИ ---
    metrics, drawdown, equity_curve = calculate_metrics(
        trades_df, initial_capital, final_equity
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
    """
    Расчет PnL (в USD) и ROI (в %) для одной сделки.
    """

    if side == "LONG":
        pnl_percent = (exit_price - entry_price) / entry_price
    elif side == "SHORT":
        pnl_percent = (entry_price - exit_price) / exit_price
    else:
        raise ValueError("Неверное направление сделки (side).")

    pnl_value = current_capital * pnl_percent * (position_size / current_capital)
    roi_percent = (pnl_value / current_capital) * 100

    final_equity = current_capital + pnl_value

    return pnl_value, roi_percent, final_equity


def generate_configs(
    parameter_space: Dict[str, Dict[str, List[Any]]], base_config: StrategyConfig
) -> Generator[StrategyConfig, None, None]:
    """Генератор, создающий комбинации StrategyConfig из пространства параметров."""

    all_params = []
    indicator_names = []

    for indicator_name, params in parameter_space.items():
        indicator_names.append(indicator_name)
        keys = list(params.keys())
        values = list(params.values())

        param_combinations = list(itertools.product(*values))

        indicator_param_sets = [dict(zip(keys, combo)) for combo in param_combinations]
        all_params.append(indicator_param_sets)

    for combo_of_indicator_sets in itertools.product(*all_params):
        new_config = copy.deepcopy(base_config)

        final_indicator_set = copy.deepcopy(base_config.indicator_set)
        for i, indicator_set in enumerate(combo_of_indicator_sets):

            # Объединяем параметры оптимизации с базовыми параметрами индикатора
            current_indicator_name = indicator_names[i]
            base_params = final_indicator_set.get(current_indicator_name, {})
            # Обновляем базовые параметры параметрами из оптимизации
            base_params.update(indicator_set)
            final_indicator_set[current_indicator_name] = base_params

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

    for run_num, current_config in enumerate(
        generate_configs(parameter_space, base_config), 1
    ):
        print(f"\n[RUN {run_num}] Тестирование: {current_config.indicator_set}")

        metrics, trades_df, _ = run_backtest(data, current_config, is_optimization=True)

        current_pnl = metrics.get("Total PnL", -np.inf)

        if current_pnl > best_total_pnl:
            best_total_pnl = current_pnl
            best_config = current_config
            best_metrics = metrics

            print(
                f"[RUN {run_num}] НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ: PnL = ${best_total_pnl:,.2f}"
            )

    execution_time = time.time() - start_time

    return best_config, best_metrics, execution_time


def main():
    # Инициализация базовых конфигураций
    strategy_config = StrategyConfig(
        initial_capital=1000.0,
        leverage=20.0,
        target_roi_percent=1.2,
        risk_roi_percent=1.0,
        # Базовый набор индикаторов для SINGLE-прогона
        indicator_set={
            "EMA_TREND": {"fast_len": 20, "slow_len": 50},
            "RSI": {"rsi_len": 10, "overbought": 70, "oversold": 30},
            "ATR_EXIT": {"atr_len": 20},
            "MACD": {"fast_len": 12, "slow_len": 26, "signal_len": 9},
            # НОВЫЕ ПОЛНОСТЬЮ РАБОЧИЕ ИНДИКАТОРЫ:
            "SWING_STRUCT": {"window": 15},  # Окно 15 свечей для обнаружения пиков
            "HTF_FILTER": {"period": "1H", "ema_len": 20},  # Тренд 1H EMA(20)
            "FIBO": {"level": 0.618},  # Уровень коррекции Фибоначчи
            # Дополнительные индикаторы (данные будут рассчитаны, но не использованы в Numba-ядре)
            "BOLLINGER_BANDS": {"bb_len": 20, "num_dev": 2.0},
            "STOCHASTIC": {"k_len": 14, "d_len": 3},
            "CCI": {"cci_len": 20},
        },
    )

    strategy_config = STRATEGY_CONFIG_MR_SCALPER

    persistence_config = PersistenceConfig(
        save_to_sqlite=False,
        save_to_csv=False,
        save_to_txt=False,
        table_name="backtest_trades",
        optimization_table_name="optimization_results",
    )

    # --- 1. ЗАГРУЗКА ДАННЫХ ---
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
            persist_results(trades_df, persistence_config)

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
                f"HTF Period: {best_config.indicator_set.get('HTF_FILTER', {}).get('period', 'N/A')}"
            )

            display_results_rich(best_metrics, pd.DataFrame(), execution_time_opt)

            # Сохраняем лучшую конфигурацию и ее метрики
            print("\n--- СОХРАНЕНИЕ ЛУЧШЕГО РЕЗУЛЬТАТА ---")
            if best_config and best_metrics:
                persist_optimization_result(
                    best_config, best_metrics, persistence_config
                )
            else:
                print(
                    "[PERSIST] Лучшая конфигурация или метрики не найдены (скорее всего, не было сделок ни в одном прогоне)."
                )

        else:
            print("\n--- ОШИБКА ОПТИМИЗАЦИИ ---\n")

    else:
        print(
            f"ОШИБКА: Неизвестный режим запуска: {RUN_MODE}. Используйте 'SINGLE' или 'OPTIMIZE'."
        )


if __name__ == "__main__":
    main()
