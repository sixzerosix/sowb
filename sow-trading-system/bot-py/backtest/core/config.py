import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd  # Для типа индекса
from pathlib import Path

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ (MOCK) ---
# Предполагаем, что файл данных лежит в той же директории
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
FILE_PATH = (
    SCRIPT_DIR
    / "backtest/data/BTC_USDT_1m_2025-08-01_00_00_00_to_2025-09-01_00_00_00.csv"
)
print(FILE_PATH)
# --- КОНФИГУРАЦИИ ---


@dataclass
class StrategyConfig:
    """
    Конфигурация торговой стратегии и параметров индикаторов.
    """

    initial_capital: float = 100.0
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


@dataclass
class OptimizationParams:
    """
    Определяет пространство поиска параметров для Optuna.
    Optuna будет искать целые числа (int) в указанных диапазонах.
    """

    # Диапазоны для EMA_TREND (целые числа)
    # Fast EMA должна быть меньше Slow EMA
    EMA_FAST_RANGE: Tuple[int, int] = (5, 50)  # [5, 50]
    EMA_SLOW_RANGE: Tuple[int, int] = (20, 150)  # [20, 150]

    # Диапазоны для RSI (целые числа)
    RSI_LEN_RANGE: Tuple[int, int] = (7, 30)  # Длина RSI
    RSI_ENTRY_RANGE: Tuple[int, int] = (
        25,
        45,
    )  # Уровень покупки (нижний, например, 30)
    RSI_EXIT_RANGE: Tuple[int, int] = (
        55,
        75,
    )  # Уровень продажи (верхний, например, 70)

    # Диапазоны для ATR (целое число, для длины)
    ATR_LEN_RANGE: Tuple[int, int] = (5, 30)

    # Общие параметры
    N_TRIALS: int = 100  # Количество попыток оптимизации

    # Метрика, которую Optuna будет пытаться оптимизировать (максимизировать/минимизировать)
    OPTIMIZATION_METRIC: str = "Total PnL"
    # Возможные варианты: 'Total PnL', 'Sharpe Ratio', 'Success Rate (%)', 'Max Drawdown (%)'
