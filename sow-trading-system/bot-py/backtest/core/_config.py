import numpy as np

# Placeholder for DataFrame type hint if needed
from typing import Dict, Any

# =========================================================================
# === КОНФИГУРАЦИОННЫЕ КЛАССЫ ===
# =========================================================================


class StrategyConfig:
    """Конфигурация параметров стратегии и ее индикаторов."""

    def __init__(
        self,
        initial_capital: float,
        leverage: float,
        target_roi_percent: float,
        risk_roi_percent: float,
        indicator_set: Dict[str, Dict[str, Any]],
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.target_roi_percent = target_roi_percent
        self.risk_roi_percent = risk_roi_percent
        self.indicator_set = indicator_set


class PersistenceConfig:
    """Конфигурация для сохранения результатов бэктеста."""

    def __init__(
        self,
        save_to_sqlite: bool,
        save_to_csv: bool,
        save_to_txt: bool,
        table_name: str,
    ):
        self.save_to_sqlite = save_to_sqlite
        self.save_to_csv = save_to_csv
        self.save_to_txt = save_to_txt
        self.table_name = table_name


# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ В main_run.py ---
# strategy_config = StrategyConfig(...)
# persistence_config = PersistenceConfig(...)
