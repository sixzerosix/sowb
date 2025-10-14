import logging
import json
from datetime import datetime
from typing import Dict, Any


def setup_logging(level=logging.INFO):
    """Настройка логирования с UTF-8"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("trading_bot.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def format_number(number: float, decimals: int = 2) -> str:
    """Форматирование числа"""
    return f"{number:.{decimals}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Расчет процентного изменения"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Безопасное деление"""
    return numerator / denominator if denominator != 0 else default


class JSONEncoder(json.JSONEncoder):
    """Кастомный JSON энкодер для datetime"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
