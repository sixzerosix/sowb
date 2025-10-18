"""
ИСПРАВЛЕННЫЙ МОДУЛЬ СОХРАНЕНИЯ РЕЗУЛЬТАТОВ

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. Добавлена поддержка Sharpe Ratio в сохранении
2. Улучшена обработка метрик
3. Исправлены импорты
"""

import pandas as pd
import sqlite3
import os
import time
from typing import Dict, Any
from pathlib import Path

# Импорты из config
from backtest.core.config import PersistenceConfig, StrategyConfig, SCRIPT_DIR


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
            trades_df.to_sql(config.table_name, conn, if_exists="append", index=False)
            conn.close()
            print(
                f"[PERSIST] Успешно сохранено в SQLite: {config.sqlite_db_name} "
                f"(Таблица: {config.table_name})"
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
            # Выбираем колонки для лога
            log_cols = []
            available_cols = trades_df.columns.tolist()

            # Определяем, какие колонки доступны
            desired_cols = [
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "pnl",
                "equity_after_trade",
            ]

            for col in desired_cols:
                if col in available_cols:
                    log_cols.append(col)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("ЛОГ СДЕЛОК БЭКТЕСТА\n")
                f.write("=" * 80 + "\n\n")
                f.write(trades_df[log_cols].to_string())
                f.write("\n\n" + "=" * 80 + "\n")

            print(f"[PERSIST] Успешно сохранено в TXT: {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в TXT: {e}")


def persist_optimization_result(
    best_config: StrategyConfig, best_metrics: Dict[str, Any], config: PersistenceConfig
):
    """
    Сохраняет лучшую конфигурацию и ее метрики в отдельный файл/таблицу.
    """

    # --- 1. Сбор данных конфигурации ---
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_capital": best_config.initial_capital,
        "leverage": best_config.leverage,
        "target_roi_percent": best_config.target_roi_percent,
        "risk_roi_percent": best_config.risk_roi_percent,
    }

    # Разворачиваем параметры индикаторов
    ema_params = best_config.indicator_set.get("EMA_TREND", {})
    record["EMA_TREND_fast_len"] = ema_params.get("fast_len")
    record["EMA_TREND_slow_len"] = ema_params.get("slow_len")

    rsi_params = best_config.indicator_set.get("RSI", {})
    record["RSI_rsi_len"] = rsi_params.get("rsi_len")
    record["RSI_overbought"] = rsi_params.get("overbought")
    record["RSI_oversold"] = rsi_params.get("oversold")

    atr_params = best_config.indicator_set.get("ATR_EXIT", {})
    record["ATR_atr_len"] = atr_params.get("atr_len")
    record["ATR_atr_multiplier"] = atr_params.get("atr_multiplier")

    # --- 2. Добавление данных метрик ---
    record["Total_Trades"] = best_metrics.get("Total Trades")
    record["Total_PnL"] = best_metrics.get("Total PnL")
    record["Final_Equity"] = best_metrics.get("Final Equity")
    record["Return_Pct"] = best_metrics.get("Return (%)")
    record["Success_Rate_Pct"] = best_metrics.get("Success Rate (%)")
    record["Profit_Factor"] = best_metrics.get("Profit Factor")
    record["Sharpe_Ratio"] = best_metrics.get("Sharpe Ratio")  # ДОБАВЛЕНО
    record["Max_Drawdown_Pct"] = best_metrics.get("Max Drawdown (%)")
    record["Avg_PnL_per_Trade"] = best_metrics.get("Avg PnL per Trade")
    record["Liquidation_Rate_Pct"] = best_metrics.get("Liquidation Rate (%)")

    # Создаем DataFrame из одной строки
    results_df = pd.DataFrame([record])

    # --- 3. Сохранение в SQLite3 ---
    if config.save_to_sqlite:
        db_path = os.path.join(SCRIPT_DIR, config.sqlite_db_name)
        try:
            conn = sqlite3.connect(db_path)
            results_df.to_sql(
                config.optimization_table_name, conn, if_exists="append", index=False
            )
            conn.close()
            print(
                f"[PERSIST] Успешно сохранено в SQLite: {config.sqlite_db_name} "
                f"(Таблица: {config.optimization_table_name})"
            )
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить результаты оптимизации в SQLite: {e}")

    # --- 4. Сохранение в CSV ---
    if config.save_to_csv:
        csv_path = os.path.join(SCRIPT_DIR, "optimization_best_results.csv")
        try:
            # Если файл существует, добавляем данные. Если нет - создаем.
            if os.path.exists(csv_path):
                results_df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                results_df.to_csv(csv_path, index=False)

            print(
                f"[PERSIST] Успешно сохранено в CSV (Лучший результат): "
                f"{os.path.basename(csv_path)}"
            )
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить результаты оптимизации в CSV: {e}")

    # --- 5. Сохранение в TXT (читаемый формат) ---
    if config.save_to_txt:
        txt_path = os.path.join(
            SCRIPT_DIR, f"optimization_result_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("РЕЗУЛЬТАТ ОПТИМИЗАЦИИ СТРАТЕГИИ\n")
                f.write("=" * 80 + "\n\n")

                f.write("ПАРАМЕТРЫ СТРАТЕГИИ:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Начальный капитал: ${best_config.initial_capital:,.2f}\n")
                f.write(f"Плечо: {best_config.leverage}x\n")
                f.write(f"Take Profit: {best_config.target_roi_percent}%\n")
                f.write(f"Stop Loss: {best_config.risk_roi_percent}%\n\n")

                f.write("ПАРАМЕТРЫ ИНДИКАТОРОВ:\n")
                f.write("-" * 80 + "\n")
                for ind_name, params in best_config.indicator_set.items():
                    f.write(f"\n{ind_name}:\n")
                    for param_name, param_value in params.items():
                        f.write(f"  - {param_name}: {param_value}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:\n")
                f.write("-" * 80 + "\n")

                for key, value in best_metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

                f.write("\n" + "=" * 80 + "\n")

            print(f"[PERSIST] Успешно сохранено в TXT: {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в TXT: {e}")
