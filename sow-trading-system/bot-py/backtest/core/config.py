import os
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
import pandas as pd  # Для типа индекса
from pathlib import Path

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ (MOCK) ---
# Предполагаем, что файл данных лежит в той же директории
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
FILE_PATH = (
    SCRIPT_DIR / "data/BTC_USDT_1m_2025-07-01_00_00_00_to_2025-09-01_00_00_00.csv"
)
print(FILE_PATH)
# --- КОНФИГУРАЦИИ ---


@dataclass
class StrategyConfig:
    """
    Конфигурация торговой стратегии и параметров индикаторов.
    """

    initial_capital: float = 1000.0
    leverage: float = 1.0
    target_roi_percent: float = 1.0
    risk_roi_percent: float = 0.5
    # НОВОЕ: Период высокого таймфрейма (например, "30T" или "1H")
    htf_period: str = "1m"  # По умолчанию - основной таймфрейм
    # Словарь для динамической настройки индикаторов
    indicator_set: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PersistenceConfig:
    """
    Конфигурация сохранения результатов бэктеста.
    """

    save_to_sqlite: bool = False
    save_to_csv: bool = True
    save_to_txt: bool = True
    sqlite_db_name: str = "backtest_results.db"
    table_name: str = "trades"
    optimization_table_name: str = "optimization_results"
    output_file_prefix: str = "trades_"
