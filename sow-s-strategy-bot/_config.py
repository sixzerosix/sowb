import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TradingConfig:
    # Bybit API
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")
    BYBIT_TESTNET: bool = os.getenv("BYBIT_TESTNET", "True").lower() == "true"

    # Telegram
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Торговые настройки
    SYMBOLS: List[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
    )
    TIMEFRAME: str = "1m"

    # Стратегия S параметры
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    RSI_PERIOD_SHORT: int = 5
    RSI_PERIOD_LONG: int = 14
    BB_PERIOD: int = 9
    BB_STD: float = 2.0
    FIBONACCI_LOOKBACK: int = 100

    # Фильтры качества
    MIN_ADX: float = 25.0
    MIN_VOLUME_RATIO: float = 1.2
    MIN_BB_WIDTH: float = 0.02
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0

    # Риск-менеджмент
    RISK_PER_TRADE: float = 0.02
    ATR_STOP_MULTIPLIER: float = 2.0
    ATR_TARGET_MULTIPLIER: float = 3.0
    MAX_POSITIONS: int = 3

    # Режим работы
    TRADING_MODE: str = "demo"

    # База данных
    DATABASE_PATH: str = "trading_bot.db"

    # Интервалы обновления (в секундах)
    DATA_COLLECTION_INTERVAL: int = 8
    SIGNAL_GENERATION_INTERVAL: int = 4
    WEBSOCKET_STREAM_INTERVAL: int = 30
    SYMBOL_LOOP_INTERVAL: int = 2


# Создание экземпляра конфигурации
config = TradingConfig()
