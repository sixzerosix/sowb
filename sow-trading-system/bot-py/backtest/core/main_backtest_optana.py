import time
import pandas as pd
import numpy as np

# Удалены itertools и copy, так как Optuna будет определять параметры
from typing import Tuple, Dict, Any, List, Generator
from numba import jit
import optuna  # <-- НОВЫЙ ИМПОРТ

# Импорт модулей
# Примечание: предполагается, что эти модули содержат необходимые классы и функции
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

# Эти конфигурации больше не нужны для оптимизации, но оставлены для режима SINGLE
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
# - "OPTIMIZE" для запуска цикла оптимизации Optuna
RUN_MODE = "OPTIMIZE"

# Какую стратегию запускать в режиме "SINGLE" (используйте вашу конфигурацию из config.py)
TARGET_CONFIG_NAME = "MR_SCALPER"


# --- ПРОСТРАНСТВО ПАРАМЕТРОВ ДЛЯ ОПТИМИЗАЦИИ OPTUNA ---
# Здесь мы задаем широкие диапазоны, которые Optuna будет исследовать.
# Мы не используем этот словарь напрямую, но он служит справочником.
PARAMETER_SEARCH_RANGES: Dict[str, Dict[str, List[Any]]] = {
    # [min, max, step] или [choice1, choice2, ...]
    "EMA_TREND": {"fast_len": [5, 20], "slow_len": [30, 60]},
    "RSI": {"rsi_len": [7, 25], "oversold": [20, 40]},
    "ATR_EXIT": {"atr_len": [10, 30]},
    "HTF_FILTER": {"period": ["15min", "30min", "1H", "2H"]},
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
    oversold: float,
    overbought: float,
    target_roi_percent: float,
    risk_roi_percent: float,
    leverage: float,
    macd_filter_pass_array: np.ndarray,  # (dtype=np.bool_)
    htf_filter_pass_array: np.ndarray,  # (dtype=np.bool_)
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

        # Фильтры теперь передаются как готовые булевы массивы
        macd_filter_pass = macd_filter_pass_array[i]
        htf_filter_pass = htf_filter_pass_array[i]

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

            # Check Trend Reversal (EMA Cross Down OR Overbought RSI OR HTF Reversal)
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
                # macd_filter_pass # <-- ВРЕМЕННО ОТКЛЮЧЕНО в исходном коде
            )

            if entry_signal:
                if trade_count < max_trades:
                    entry_price = current_close
                    open_idx = i
                    is_long = True

                    # Расчет SL и TP в цене на основе ATR и % риска/цели
                    atr_distance = atr_val[i]  # Берем ATR на текущей свече

                    # target_roi_percent и risk_roi_percent теперь множители ATR
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
        # print(
        #     "--- WARNING: Недостаточно данных после расчета индикаторов. Пропускаем прогон."
        # )
        return (
            {
                "Total Trades": 0,
                "Total PnL": -1000.0,  # Отрицательное PnL для Optuna, чтобы избежать выбора недействительных прогонов
                "Final Equity": initial_capital,
                "Return (%)": 0.0,
            },
            pd.DataFrame(),
            df_with_indicators,
        )

    # --- 2. ПОДГОТОВКА ДАННЫХ ДЛЯ NUMBA (ТОЛЬКО NUMPY МАССИВЫ) ---
    closes = df_with_indicators["close"].values

    # EMA/RSI (предполагаем, что они базовые)
    ema_fast = df_with_indicators.get("ema_fast", np.full(len(closes), 0.0)).values
    ema_slow = df_with_indicators.get("ema_slow", np.full(len(closes), 0.0)).values
    rsi_val = df_with_indicators.get(
        "rsi_val", np.full(len(closes), 50.0)
    ).values  # 50 = нейтральный RSI

    # ATR (он нужен для SL/TP, поэтому используем дефолт, если его нет)
    atr_val = df_with_indicators.get("atr_val", np.full(len(closes), 0.01)).values

    # Параметры из Config
    oversold = config.indicator_set.get("RSI", {}).get("oversold", 30.0)
    overbought = config.indicator_set.get("RSI", {}).get("overbought", 70.0)
    target_roi_percent = config.target_roi_percent
    risk_roi_percent = config.risk_roi_percent

    # --- 2b. ПОДГОТОВКА ФИЛЬТРОВ (ДИНАМИЧЕСКИ) ---

    # Фильтр MACD: Активен, только если MACD есть в конфиге
    if "MACD" in config.indicator_set:
        if "macd_hist" in df_with_indicators.columns:
            macd_filter_pass_array = df_with_indicators["macd_hist"].values > 0.0
            if not is_optimization:
                print("--- DEBUG: Фильтр MACD (Hist > 0) АКТИВИРОВАН.")
        else:
            if not is_optimization:
                print(
                    "--- WARNING: MACD в конфиге, но колонка 'macd_hist' не найдена. Фильтр MACD отключен."
                )
            macd_filter_pass_array = np.full(len(closes), True, dtype=np.bool_)
    else:
        # Если MACD нет в конфиге, фильтр всегда True (пропускает)
        if not is_optimization:
            print("--- DEBUG: MACD не в конфиге. Фильтр MACD (логика) отключен.")
        macd_filter_pass_array = np.full(len(closes), True, dtype=np.bool_)

    # Фильтр HTF: Активен, только если HTF_FILTER есть в конфиге
    if "HTF_FILTER" in config.indicator_set:
        if "htf_trend_up" in df_with_indicators.columns:
            htf_filter_pass_array = df_with_indicators["htf_trend_up"].values
            if not is_optimization:
                print("--- DEBUG: Фильтр HTF (Trend UP) АКТИВИРОВАН.")
        else:
            if not is_optimization:
                print(
                    "--- WARNING: HTF_FILTER в конфиге, но 'htf_trend_up' не найдена. Фильтр HTF отключен."
                )
            htf_filter_pass_array = np.full(len(closes), True, dtype=np.bool_)
    else:
        # Если HTF_FILTER нет в конфиге, фильтр всегда True (пропускает)
        if not is_optimization:
            print("--- DEBUG: HTF_FILTER не в конфиге. Фильтр HTF (логика) отключен.")
        htf_filter_pass_array = np.full(len(closes), True, dtype=np.bool_)

    # --- 3. ВЫЗОВ NUMBA-КОМПИЛИРОВАННОГО ЯДРА ---
    trades_array, final_equity = numba_backtest_core(
        closes,
        ema_fast,
        ema_slow,
        rsi_val,
        atr_val,
        oversold,
        overbought,
        target_roi_percent,
        risk_roi_percent,
        config.leverage,
        macd_filter_pass_array,  # Новый параметр
        htf_filter_pass_array,  # Новый параметр
    )

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
        # В режиме оптимизации не выводим предупреждения, чтобы не загромождать консоль
        if not is_optimization:
            print("--- WARNING: В данном прогоне сделки не совершались.")
        metrics = {
            "Total Trades": 0,
            "Total PnL": -1000.0,  # Отрицательное PnL для Optuna
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
    # Оставляем полную логику возврата, но Optuna будет использовать только 'Total PnL'
    return metrics, trades_df, df_with_indicators


# =========================================================================
# === МОДУЛЬ 5: OPTUNA ОПТИМИЗАЦИЯ ===
# =========================================================================


def objective(
    trial: optuna.Trial, data: pd.DataFrame, base_config: StrategyConfig
) -> float:
    """
    Целевая функция Optuna: определяет пространство поиска и запускает бэктест.
    """

    # 1. Определение пространства поиска параметров с помощью Optuna Trial

    # EMA Trend
    fast_len = trial.suggest_int("ema_fast", 5, 20)
    slow_len = trial.suggest_int("ema_slow", 30, 60)

    # RSI
    rsi_len = trial.suggest_int("rsi_len", 7, 25)
    oversold = trial.suggest_int("oversold", 20, 40)

    # ATR Exit
    atr_len = trial.suggest_int("atr_len", 10, 30)

    # HTF Filter
    htf_period = trial.suggest_categorical("htf_period", ["15min", "30min", "1H", "2H"])

    # Убедимся, что fast_len < slow_len
    if fast_len >= slow_len:
        # Использование Trial Pruning для отбрасывания нелогичных комбинаций
        raise optuna.exceptions.TrialPruned()

    # 2. Создание новой StrategyConfig
    # Копируем базовую конфигурацию, чтобы сохранить параметры, не участвующие в оптимизации
    # Используем deepcopy, чтобы избежать изменения глобальной STRATEGY_CONFIG
    new_config = StrategyConfig(
        initial_capital=base_config.initial_capital,
        leverage=base_config.leverage,
        target_roi_percent=base_config.target_roi_percent,
        risk_roi_percent=base_config.risk_roi_percent,
        indicator_set={
            # Переопределяем или создаем наборы индикаторов
            "EMA_TREND": {"fast_len": fast_len, "slow_len": slow_len},
            "RSI": {
                "rsi_len": rsi_len,
                "overbought": 70.0,
                "oversold": float(oversold),
            },
            "ATR_EXIT": {"atr_len": atr_len},
            "HTF_FILTER": {"period": htf_period, "ema_len": 20},
            # Добавляем остальные индикаторы из базы, если они есть (MACD, SWING_STRUCT и т.д.)
            # Здесь предполагаем, что если MACD не оптимизируется, мы берем его из базы
            **{
                k: v
                for k, v in base_config.indicator_set.items()
                if k not in ["EMA_TREND", "RSI", "ATR_EXIT", "HTF_FILTER"]
            },
        },
    )

    # 3. Сохранение параметров в атрибутах Trial (для последующего извлечения лучшей конфигурации)
    # Нам нужно сохранить всю конфигурацию
    trial.set_user_attr("optimized_config_params", new_config.indicator_set)

    # 4. Запуск бэктеста
    metrics, _, _ = run_backtest(data, new_config, is_optimization=True)

    # 5. Возвращаем целевую метрику (например, Total PnL)
    return metrics.get("Total PnL", -np.inf)


def run_optuna_optimization(
    data: pd.DataFrame,
    base_config: StrategyConfig,
    persistence_config: PersistenceConfig,
    n_trials: int = 100,  # Количество прогонов Optuna
) -> Tuple[StrategyConfig | None, Dict[str, Any] | None, float]:
    """Запускает цикл оптимизации параметров стратегии с использованием Optuna."""

    start_time = time.time()

    # 1. Создание Study (цель - максимизировать Total PnL)
    study = optuna.create_study(
        direction="maximize",
        study_name="Backtest_Strategy_Optimization",
    )

    # 2. Запуск оптимизации
    print(f"\n--- Optuna: Начинаем оптимизацию (N={n_trials} прогонов) ---")

    # Optuna запускает целевую функцию n_trials раз
    study.optimize(
        lambda trial: objective(trial, data, base_config),
        n_trials=n_trials,
        # Мы не используем Pruners, но их можно добавить для ускорения.
    )

    execution_time = time.time() - start_time

    # 3. Извлечение лучшего результата
    try:
        best_trial = study.best_trial
        best_metrics, _, _ = run_backtest(
            data, base_config
        )  # Перезапуск для получения метрик и DF

        # Получаем оптимизированные параметры из атрибутов
        best_params_dict = best_trial.user_attrs["optimized_config_params"]

        # Создаем лучшую конфигурацию
        best_config = StrategyConfig(
            initial_capital=base_config.initial_capital,
            leverage=base_config.leverage,
            target_roi_percent=base_config.target_roi_percent,
            risk_roi_percent=base_config.risk_roi_percent,
            indicator_set=best_params_dict,
        )

        # Перезапускаем бэктест на лучшей конфигурации, чтобы получить полные метрики и trades_df
        # Это также генерирует индикаторы для persist_optimization_result, если нужно
        best_metrics, trades_df, _ = run_backtest(
            data, best_config, is_optimization=False
        )

        # Добавляем PnL из лучшего прогона Optuna, так как metrics из run_backtest может быть неполным
        if (
            "Total PnL" not in best_metrics
            or best_metrics["Total PnL"] < best_trial.value
        ):
            best_metrics["Total PnL"] = best_trial.value

        print("\n--- Optuna: Лучший результат найден ---")
        print(f"PnL: ${best_trial.value:,.2f}")
        print(f"Параметры: {best_trial.params}")

        return best_config, best_metrics, execution_time

    except ValueError:
        print("--- Optuna: Не удалось найти лучший прогон (нет успешных сделок).")
        return None, None, execution_time
    except Exception as e:
        print(f"--- Optuna: Произошла ошибка при извлечении лучшего прогона: {e}")
        return None, None, execution_time


def main():
    # Инициализация базовых конфигураций
    # Используем STRATEGY_CONFIG_MR_SCALPER как базовую, но Optuna будет ее переопределять.
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
        print(f"\n--- ЗАПУСК В РЕЖИМЕ '{RUN_MODE}' (Optuna) ---")

        # Optuna запускается с базовой конфигурацией,
        # которая служит шаблоном для параметров, не участвующих в оптимизации.
        best_config, best_metrics, execution_time_opt = run_optuna_optimization(
            data, strategy_config, persistence_config, n_trials=50
        )  # Установил 50 прогонов для примера

        if best_config and best_metrics:
            print("\n--- ЛУЧШАЯ КОНФИГУРАЦИЯ (BEST OPTUNA TRIAL) ---")

            # Отображаем только оптимизированные параметры для примера
            optimized_params = best_config.indicator_set

            print(f"PnL: ${best_metrics['Total PnL']:,.2f}")
            print(f"Return (%): {best_metrics.get('Return (%)', 0.0):.2f}%")
            print(f"Max Drawdown: {best_metrics.get('Max Drawdown (%)', 0.0):.2f}%")
            print(
                f"EMA Fast: {optimized_params.get('EMA_TREND', {}).get('fast_len', 'N/A')}"
            )
            print(
                f"EMA Slow: {optimized_params.get('EMA_TREND', {}).get('slow_len', 'N/A')}"
            )
            print(f"RSI Len: {optimized_params.get('RSI', {}).get('rsi_len', 'N/A')}")
            print(
                f"HTF Period: {optimized_params.get('HTF_FILTER', {}).get('period', 'N/A')}"
            )

            display_results_rich(best_metrics, pd.DataFrame(), execution_time_opt)

            # Сохраняем лучшую конфигурацию и ее метрики
            print("\n--- СОХРАНЕНИЕ ЛУЧШЕГО РЕЗУЛЬТАТА ---")
            persist_optimization_result(best_config, best_metrics, persistence_config)

        else:
            print("\n--- ОШИБКА ОПТИМИЗАЦИИ (Optuna) ---")
            print(
                "Не удалось найти лучшую конфигурацию. Все прогоны, вероятно, не смогли совершить сделки или были отброшены."
            )

    else:
        print(
            f"ОШИБКА: Неизвестный режим запуска: {RUN_MODE}. Используйте 'SINGLE' или 'OPTIMIZE'."
        )


if __name__ == "__main__":
    main()
