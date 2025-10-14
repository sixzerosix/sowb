import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class HighFrequencyTradingConfig:
    # Bybit API
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")
    BYBIT_TESTNET: bool = False

    # Telegram
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # 🚀 ВЫСОКОЧАСТОТНЫЕ настройки
    SYMBOLS: List[str] = field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "ADA/USDT",
            "MNT/USDT",
            "RATS/USDT",
            "SNX/USDT",
        ]
    )  # Меньше символов = больше фокуса
    TIMEFRAME: str = "1m"  # 1 минутные свечи!
    ANALYSIS_TIMEFRAME: str = "1m"  # Анализ на 1м

    # ⚡ Скальпинг параметры стратегии S
    EMA_FAST: int = 5  # Было 9, теперь 5 - быстрее реакция
    EMA_SLOW: int = 13  # Было 21, теперь 13 - быстрее сигналы
    RSI_PERIOD_SHORT: int = 3  # Было 5
    RSI_PERIOD_LONG: int = 7  # Было 14
    BB_PERIOD: int = 7  # Было 9
    BB_STD: float = 1.8  # Было 2.0 - более чувствительные полосы
    FIBONACCI_LOOKBACK: int = 50  # Было 100 - меньше истории

    # 🎯 Агрессивные фильтры для скальпинга
    MIN_ADX: float = 15.0  # Было 25.0 - менее строгий фильтр
    MIN_VOLUME_RATIO: float = 1.1  # Было 1.2
    MIN_BB_WIDTH: float = 0.008  # Было 0.02 - более узкие полосы
    RSI_OVERSOLD: float = 25.0  # Было 30.0 - более агрессивно
    RSI_OVERBOUGHT: float = 75.0  # Было 70.0 - более агрессивно

    # 💰 ПЛЕЧО И РИСК-МЕНЕДЖМЕНТ
    LEVERAGE: int = 10  # 10x плечо!
    RISK_PER_TRADE: float = 0.01  # 1% риска на сделку (с плечом = 10% от баланса)
    ATR_STOP_MULTIPLIER: float = 1.5  # Было 2.0 - более близкий стоп
    ATR_TARGET_MULTIPLIER: float = 2.0  # Было 3.0 - более близкая цель
    MAX_POSITIONS: int = 2  # Максимум 2 позиции одновременно

    # ⚡ БЫСТРЫЕ ИНТЕРВАЛЫ
    DATA_UPDATE_INTERVAL: int = 10  # 10 секунд обновление данных!
    SIGNAL_CHECK_INTERVAL: int = 5  # 5 секунд проверка сигналов!
    WEBSOCKET_UPDATE_INTERVAL: int = 1  # 1 секунда WebSocket

    # 🎯 СКАЛЬПИНГ НАСТРОЙКИ
    MAX_TRADE_DURATION: int = 900  # 15 минут максимум в сделке
    PREFERRED_TRADE_DURATION: int = 300  # 5 минут предпочтительно
    MIN_PROFIT_POINTS: float = 0.1  # Минимум 0.1% прибыли
    QUICK_EXIT_THRESHOLD: float = 0.05  # Быстрый выход при 0.05% убытке

    # 📊 УВЕРЕННОСТЬ ДЛЯ СКАЛЬПИНГА
    MIN_CONFIDENCE_FOR_TRADE: float = 0.6  # Было 0.7 - более агрессивно
    MIN_CONFIDENCE_FOR_LEVERAGE: float = 0.8  # Плечо только при высокой уверенности

    # Режим работы
    TRADING_MODE: str = "demo"
    DATABASE_PATH: str = "scalping_bot.db"

    # Интервалы обновления (в секундах)
    DATA_COLLECTION_INTERVAL: int = 8
    SIGNAL_GENERATION_INTERVAL: int = 4
    WEBSOCKET_STREAM_INTERVAL: int = 30
    SYMBOL_LOOP_INTERVAL: int = 2


# Создание экземпляра конфигурации
config = HighFrequencyTradingConfig()
