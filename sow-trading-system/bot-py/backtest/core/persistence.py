import pandas as pd
import sqlite3
import os
import time
from backtest.core.config import PersistenceConfig, SCRIPT_DIR

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
