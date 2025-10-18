import pandas as pd
import sqlite3
import os
import time
from typing import Dict, Any, List


# --- ЗАГЛУШКИ ДЛЯ КЛАССОВ (В РЕАЛЬНОМ ПРОЕКТЕ ИМПОРТИРУЙТЕ ИЗ backtest.core.config) ---
class PersistenceConfig:
    def __init__(
        self,
        save_to_sqlite: bool,
        save_to_csv: bool,
        save_to_txt: bool,
        table_name: str,
        optimization_table_name: str = "optimization_results",
        sqlite_db_name: str = "backtest_results.db",
        output_file_prefix: str = "trades_",
    ):
        self.save_to_sqlite = save_to_sqlite
        self.save_to_csv = save_to_csv
        self.save_to_txt = save_to_txt
        self.table_name = table_name
        self.optimization_table_name = optimization_table_name  # Новое имя таблицы
        self.sqlite_db_name = sqlite_db_name
        self.output_file_prefix = output_file_prefix


class StrategyConfig:
    def __init__(
        self,
        initial_capital: float,
        leverage: float,
        target_roi_percent: float,
        risk_roi_percent: float,
        indicator_set: Dict,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.target_roi_percent = target_roi_percent
        self.risk_roi_percent = risk_roi_percent
        self.indicator_set = indicator_set


# Предполагается, что SCRIPT_DIR определен в backtest.core.config или в другом месте
# Временно определяем заглушку, чтобы файл компилировался
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# -------------------------------------------------------------------------------------

# =========================================================================
# === МОДУЛЬ 5: ПЕРСИСТЕНТНОСТЬ (SQLite, CSV, TXT) ===
# =========================================================================


def persist_results(trades_df: pd.DataFrame, config: PersistenceConfig):
    """Сохраняет результаты сделок в указанные форматы."""
    if trades_df.empty:
        print("[PERSIST] Нет сделок для сохранения.")
        return

    file_prefix = config.output_file_prefix

    # --- 1. Сохранение в SQLite3 ---
    if config.save_to_sqlite:
        db_path = os.path.join(SCRIPT_DIR, config.sqlite_db_name)
        try:
            conn = sqlite3.connect(db_path)
            # Сохраняем сделку вместе с метадандами (имя стратегии, дата и т.д.)
            trades_df.to_sql(config.table_name, conn, if_exists="append", index=False)
            conn.close()
            print(
                f"[PERSIST] Успешно сохранено в SQLite: {config.sqlite_db_name} (Таблица: {config.table_name})"
            )
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в SQLite: {e}")

    # --- 2. Сохранение в CSV ---
    if config.save_to_csv:
        csv_path = os.path.join(
            SCRIPT_DIR, f"{file_prefix}{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        try:
            trades_df.to_csv(csv_path, index=False)
            print(f"[PERSIST] Успешно сохранено в CSV: {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в CSV: {e}")

    # --- 3. Сохранение в TXT (Простой лог) ---
    if config.save_to_txt:
        txt_path = os.path.join(
            SCRIPT_DIR, f"{file_prefix}{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        try:
            log_cols = [
                "entry_time",
                "side",
                "entry_price",
                "exit_price",
                "pnl",
                "close_reason",
                "equity_after_trade",
            ]
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("--- ЛОГ СДЕЛОК БЭКТЕСТА ---\n")
                f.write(trades_df[log_cols].to_string())
            print(f"[PERSIST] Успешно сохранено в TXT: {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в TXT: {e}")


def persist_optimization_result(
    best_config: StrategyConfig, best_metrics: Dict[str, Any], config: PersistenceConfig
):
    """
    Сохраняет лучшую конфигурацию и ее метрики в отдельный файл/таблицу.

    Создает плоский словарь из StrategyConfig и best_metrics.
    """

    # --- 1. Сбор данных конфигурации (Разворачиваем StrategyConfig) ---
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_capital": best_config.initial_capital,
        "leverage": best_config.leverage,
        "target_roi_percent": best_config.target_roi_percent,
        "risk_roi_percent": best_config.risk_roi_percent,
    }

    # Разворачиваем параметры индикаторов (берем только оптимизированные для примера)
    # Здесь можно добавить логику для всех индикаторов, но для примера берем EMA и RSI
    ema_params = best_config.indicator_set.get("EMA_TREND", {})
    record["EMA_TREND_fast_len"] = ema_params.get("fast_len")
    record["EMA_TREND_slow_len"] = ema_params.get("slow_len")

    rsi_params = best_config.indicator_set.get("RSI", {})
    record["RSI_rsi_len"] = rsi_params.get("rsi_len")

    # Добавьте другие важные параметры, например:
    atr_params = best_config.indicator_set.get("ATR_EXIT", {})
    record["ATR_min_multiplier"] = atr_params.get("min_multiplier")

    # --- 2. Добавление данных метрик (Разворачиваем best_metrics) ---

    # Используем только ключевые метрики, чтобы таблица была читаемой
    record["Total_Return_Pct"] = best_metrics.get("Total Return (%)")
    record["Final_Equity"] = best_metrics.get("Финальный капитал")
    record["Max_Drawdown_Pct"] = best_metrics.get("Максимальная просадка (MDD)")
    record["Total_Trades"] = best_metrics.get("Общее количество сделок")
    record["Win_Rate_Pct"] = best_metrics.get("Процент прибыльных сделок")
    record["Execution_Time_sec"] = best_metrics.get("Время выполнения (Numba)")

    # Создаем DataFrame из одной строки
    results_df = pd.DataFrame([record])

    # --- 3. Сохранение в SQLite3 ---
    if config.save_to_sqlite:
        db_path = os.path.join(SCRIPT_DIR, config.sqlite_db_name)
        try:
            conn = sqlite3.connect(db_path)
            # Используем отдельное имя таблицы для результатов оптимизации
            results_df.to_sql(
                config.optimization_table_name, conn, if_exists="append", index=False
            )
            conn.close()
            print(
                f"[PERSIST] Успешно сохранено в SQLite: {config.sqlite_db_name} (Таблица: {config.optimization_table_name})"
            )
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить результаты оптимизации в SQLite: {e}")

    # --- 4. Сохранение в CSV (Отдельный файл для истории лучших прогонов) ---
    if config.save_to_csv:
        csv_path = os.path.join(SCRIPT_DIR, "optimization_best_results.csv")
        try:
            # Если файл существует, добавляем данные. Если нет - создаем.
            if os.path.exists(csv_path):
                results_df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                results_df.to_csv(csv_path, index=False)
            print(
                f"[PERSIST] Успешно сохранено в CSV (Лучший результат): {os.path.basename(csv_path)}"
            )
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить результаты оптимизации в CSV: {e}")
