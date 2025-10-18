import time
import pandas as pd
import numpy as np
from numba import njit  # Используем njit для имитации ускоренного ядра

# Импорт модулей
from backtest.core.config import StrategyConfig, PersistenceConfig, FILE_PATH
from backtest.core.data_loader import load_data
from backtest.core.indicators import calculate_strategy_indicators
from backtest.core.persistence import persist_results
from backtest.core.analysis import calculate_metrics, display_results_rich, plot_results


# =========================================================================
# === МОДУЛЬ 3: ЯДРО БЭКТЕСТЕРА (NUMBA SIMULATION MOCK) ===
# =========================================================================


# @njit # Оставляем закомментированным, так как njit требует чистых numba-типов
def run_backtest(df_processed: pd.DataFrame, config: StrategyConfig):
    """
    ЗАГЛУШКА ЯДРА БЭКТЕСТЕРА (Numba-ядро).
    Она имитирует логику открытия/закрытия позиций на основе индикаторов
    и возвращает DataFrame с результатами сделок.

    В реальной версии здесь был бы чистый numpy/numba-код.
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
    target_pnl_abs = initial_capital * config.target_roi_percent / 100 * leverage
    risk_pnl_abs = initial_capital * config.risk_roi_percent / 100 * leverage

    current_equity = initial_capital
    trades = []

    # Имитация входа в позицию (Long, если EMA Fast > EMA Slow)
    in_position = False

    for i in range(len(df_processed)):
        current_price = close_prices[i]

        # Имитация сигнала на вход (Простое пересечение EMA)
        long_condition = (
            fast_ema[i] > slow_ema[i]
            and fast_ema[i - 1] <= slow_ema[i - 1]
            and not in_position
        )

        if long_condition and i > 1:
            entry_price = current_price

            # Расчет Take Profit и Stop Loss на основе ATR (для демонстрации)
            # В реальной стратегии Fibo Scalper здесь использовались бы уровни Фибо
            tp_abs = atr_val[i] * df_processed.get("atr_multiplier", 2.0).values[i]
            sl_abs = atr_val[i] * df_processed.get("atr_multiplier", 1.0).values[i]

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
            # print(f"--- MOCK TRADE OPEN: {trade_data['side']} at {entry_price:.2f}")

        # Имитация закрытия позиции (Take Profit или Stop Loss)
        if in_position:
            # Моделируем закрытие по следующей свече (простой вариант)
            if i + 1 < len(df_processed):
                next_close = close_prices[i + 1]
                pnl_ratio = (next_close - trade_data["entry_price"]) / trade_data[
                    "entry_price"
                ]

                # Имитация PNL
                pnl = pnl_ratio * trade_data["position_size"]

                # Имитация причины закрытия (случайно)
                close_reason = "TP/SL Mock" if np.random.rand() > 0.3 else "Signal Mock"

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

    trades_df = pd.DataFrame(trades)
    final_equity = current_equity

    # Добавление пустых столбцов для совместимости с analysis.py, если их нет
    if trades_df.empty:
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
# === ОСНОВНОЙ СКРИПТ: ИНТЕГРАЦИЯ ===
# =========================================================================

if __name__ == "__main__":

    print("--- ЗАПУСК БЭКТЕСТЕРА ---")

    # 1. ИНИЦИАЛИЗАЦИЯ КОНФИГУРАЦИЙ
    strategy_config = StrategyConfig(
        initial_capital=100.0,
        leverage=20.0,
        target_roi_percent=17.0,  # Немного уменьшим для реалистичности
        risk_roi_percent=10.0,
        # Переопределяем параметры индикаторов через новый модульный словарь:
        indicator_set={
            "EMA_TREND": {"fast_len": 12, "slow_len": 26},  # MACD-подобные настройки
            "ATR_EXIT": {"atr_len": 20, "min_multiplier": 2.0},
            "SWING_STRUCT": {"window": 40},
            "HTF_FILTER": {"period": "15min", "ema_fast_len": 12, "ema_slow_len": 26},
            "FIBO": {"levels": np.array([0.382, 0.618], dtype=np.float64)},
            # Активируем новые индикаторы
            "RSI": {"rsi_len": 7},
            "MACD": {"fast_p": 12, "slow_p": 26, "signal_p": 9},
            "BOLLINGER_BANDS": {"period": 20, "dev_up": 2.5, "dev_dn": 2.5},
            "VOLUME": {},
        },
    )

    persistence_config = PersistenceConfig(
        save_to_sqlite=False,
        save_to_csv=True,
        save_to_txt=False,
        table_name="fibo_scalper_run_2",
    )

    print("\n[INFO] Начинаем загрузку данных...")

    # 2. ЗАГРУЗКА И РАСЧЕТ ИНДИКАТОРОВ
    df_raw = load_data(str(FILE_PATH))

    # Вызов модульной функции расчета индикаторов
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
