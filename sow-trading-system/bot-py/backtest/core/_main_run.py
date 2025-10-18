import time
import numpy as np

# Импортируем все необходимое из модулей
from backtest.core.config import StrategyConfig, PersistenceConfig
from backtest.core._strategy_engine import (
    load_data,
    calculate_strategy_indicators,
    run_backtest,
    calculate_metrics,
    display_results_rich,
    persist_results,
    plot_results,
)

# Глобальные константы (или их можно перенести в отдельный config.py)
FILE_PATH = "data/BTCUSDT_5min.csv"

# =========================================================================
# === ОСНОВНОЙ СКРИПТ: ИНТЕГРАЦИЯ И ОРКЕСТРАЦИЯ ===
# =========================================================================


def main():
    """Главная функция для оркестрации процесса бэктестинга."""
    print("--- ЗАПУСК БЭКТЕСТЕРА ---")

    # 1. ИНИЦИАЛИЗАЦИЯ КОНФИГУРАЦИЙ
    strategy_config = StrategyConfig(
        initial_capital=100.0,
        leverage=20.0,
        target_roi_percent=2.0,
        risk_roi_percent=1.0,
        # Переопределяем параметры индикаторов через новый модульный словарь:
        indicator_set={
            "EMA_TREND": {"fast_len": 12, "slow_len": 26},
            "ATR_EXIT": {"atr_len": 20, "min_multiplier": 2.0},
            "SWING_STRUCT": {"window": 40},
            "HTF_FILTER": {"period": "15min", "ema_fast_len": 12, "ema_slow_len": 26},
            "FIBO": {"levels": np.array([0.382, 0.618], dtype=np.float64)},
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

    # 2. ЗАГРУЗКА И РАСЧЕТ ИНДИКАТОРОВ (Вызовы из strategy_engine)
    df_raw = load_data(FILE_PATH)
    df_processed = calculate_strategy_indicators(df_raw, strategy_config)

    start_time = time.time()

    # 3. ЗАПУСК БЭКТЕСТА (Вызов из strategy_engine)
    trades_df, final_equity = run_backtest(df_processed, strategy_config)

    end_time = time.time()
    execution_time = end_time - start_time

    # 4. АНАЛИЗ И ОТЧЕТНОСТЬ (Вызовы из strategy_engine)
    metrics, drawdown_for_plot, equity_curve = calculate_metrics(
        trades_df, strategy_config.initial_capital, final_equity
    )

    display_results_rich(metrics, trades_df, execution_time)

    # 5. ПЕРСИСТЕНТНОСТЬ (Вызов из strategy_engine)
    persist_results(trades_df, persistence_config)

    # 6. ГРАФИКИ (Вызов из strategy_engine)
    # plot_results(
    #     trades_df, equity_curve, drawdown_for_plot, strategy_config.initial_capital
    # )

    print(
        "\n[INFO] Для отображения графиков раскомментируйте строку 'plot_results(...)' в конце скрипта."
    )
    print("--- КОНЕЦ СИМУЛЯЦИИ ---")


if __name__ == "__main__":
    main()
