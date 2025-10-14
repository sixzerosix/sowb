import asyncio
import logging
from datetime import datetime
import signal
import sys

from config import config
from database import DatabaseManager
from _strategy import EnhancedStrategyS
from exchange import BybitExchange
from telegram_bot import TelegramBot
from risk_manager import RiskManager
from utils import setup_logging


class TradingBotS:
    def __init__(self):
        self.config = config
        self.db_manager = None
        self.strategy = None
        self.exchange = None
        self.telegram_bot = None
        self.risk_manager = None
        self.running = False

    async def initialize(self):
        """Инициализация всех компонентов бота"""
        try:
            logging.info("Инициализация торгового бота 'Стратегия S'...")

            # Инициализация базы данных
            self.db_manager = DatabaseManager(self.config.DATABASE_PATH)
            await self.db_manager.init_database()
            logging.info("База данных инициализирована")

            # Инициализация биржи
            self.exchange = BybitExchange(self.config)
            await self.exchange.initialize()
            logging.info("Подключение к Bybit установлено")

            # Инициализация стратегии
            self.strategy = EnhancedStrategyS(self.config, self.db_manager)
            logging.info("Стратегия инициализирована")

            # Инициализация риск-менеджера
            self.risk_manager = RiskManager(self.config, self.db_manager)
            logging.info("Риск-менеджер инициализирован")

            # Инициализация Telegram бота
            if self.config.TELEGRAM_TOKEN:
                self.telegram_bot = TelegramBot(
                    self.config, self.db_manager, self.strategy, self.exchange
                )
                logging.info("Telegram бот инициализирован")

            logging.info("Все компоненты успешно инициализированы")

        except Exception as e:
            logging.error(f"Ошибка инициализации: {e}")
            raise

    async def start_data_collection(self):
        """Запуск сбора данных"""
        while self.running:
            try:
                # Получение OHLCV данных для всех символов
                ohlcv_data = await self.exchange.fetch_ohlcv_batch(
                    self.config.SYMBOLS, self.config.TIMEFRAME, 100
                )

                # Сохранение данных в БД
                for symbol, data in ohlcv_data.items():
                    if data:
                        formatted_data = []
                        for candle in data:
                            formatted_data.append(
                                {
                                    "timestamp": candle[0],
                                    "symbol": symbol,
                                    "timeframe": self.config.TIMEFRAME,
                                    "open": candle[1],
                                    "high": candle[2],
                                    "low": candle[3],
                                    "close": candle[4],
                                    "volume": candle[5],
                                }
                            )

                        await self.db_manager.insert_ohlcv_batch(formatted_data)

                logging.info(f"Данные обновлены для {len(ohlcv_data)} символов")

                # Пауза между обновлениями
                await asyncio.sleep(300)  # 5 минут

            except Exception as e:
                logging.error(f"Ошибка сбора данных: {e}")
                await asyncio.sleep(60)

    async def start_signal_generation(self):
        """Запуск генерации торговых сигналов"""
        while self.running:
            try:
                for symbol in self.config.SYMBOLS:
                    # Генерация сигнала
                    signal = await self.strategy.analyze_symbol(symbol)

                    if signal.action != "HOLD":
                        logging.info(
                            f"Сигнал {symbol}: {signal.action} at ${signal.price:.4f} "
                            f"(confidence: {signal.confidence:.2%})"
                        )

                        # Отправка уведомления в Telegram
                        if self.telegram_bot:
                            await self.telegram_bot.send_signal_notification(signal)

                        # Выполнение сделки если уверенность высокая
                        if signal.confidence > 0.7:
                            await self.execute_signal(signal)

                    await asyncio.sleep(2)  # Пауза между символами

                # Пауза между циклами анализа
                await asyncio.sleep(60)  # 1 минута

            except Exception as e:
                logging.error(f"Ошибка генерации сигналов: {e}")
                await asyncio.sleep(30)

    async def execute_signal(self, signal):
        """Выполнение торгового сигнала"""
        try:
            # Проверка возможности открытия позиции
            if not await self.risk_manager.can_open_position(signal.symbol):
                logging.warning(f"Позицию по {signal.symbol} открыть нельзя")
                return

            # Получение информации об аккаунте
            account_info = await self.exchange.get_account_info()
            balance = account_info.get("balance", {}).get("USDT", {}).get("free", 0)

            if balance < 10:  # Минимальный баланс
                logging.warning(f"Недостаточный баланс: ${balance}")
                return

            # Расчет размера позиции
            position_size = await self.risk_manager.calculate_position_size(
                signal.symbol, signal.price, signal.stop_loss, balance
            )

            if position_size <= 0:
                logging.warning(f"Размер позиции для {signal.symbol} равен 0")
                return

            # Размещение ордера
            order = await self.exchange.place_order(
                symbol=signal.symbol,
                side=signal.action.lower(),
                amount=position_size,
                price=None,  # Market order
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            if order:
                # Сохранение сделки в БД
                await self.save_trade_to_db(signal, order, position_size)

                # Добавление позиции в риск-менеджер
                self.risk_manager.add_position(
                    signal.symbol, signal.action, position_size, signal.price
                )

                logging.info(
                    f"Ордер выполнен: {signal.symbol} {signal.action} "
                    f"{position_size:.6f} at ${signal.price:.4f}"
                )

                # Уведомление в Telegram
                if self.telegram_bot:
                    await self.telegram_bot.bot.send_message(
                        chat_id=self.config.TELEGRAM_CHAT_ID,
                        text=f"✅ <b>Ордер выполнен</b>\n\n"
                        f"Символ: {signal.symbol}\n"
                        f"Действие: {signal.action}\n"
                        f"Размер: {position_size:.6f}\n"
                        f"Цена: ${signal.price:.4f}\n"
                        f"Режим: {self.config.TRADING_MODE}",
                        parse_mode="HTML",
                    )

        except Exception as e:
            logging.error(f"Ошибка выполнения сигнала {signal.symbol}: {e}")

    async def save_trade_to_db(self, signal, order, position_size):
        """Сохранение сделки в базу данных"""
        try:
            async with self.db_manager.db_path as db:
                await db.execute(
                    """
                    INSERT INTO trades (
                        timestamp, symbol, side, amount, price, order_id,
                        entry_price, status, mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        signal.timestamp,
                        signal.symbol,
                        signal.action,
                        position_size,
                        signal.price,
                        order.get("id"),
                        signal.price,
                        "open",
                        self.config.TRADING_MODE,
                    ),
                )
                await db.commit()

        except Exception as e:
            logging.error(f"Ошибка сохранения сделки в БД: {e}")

    async def start_websocket_streams(self):
        """Запуск WebSocket потоков"""
        try:
            await self.exchange.start_websocket_streams(
                self.config.SYMBOLS, self.handle_websocket_data
            )
        except Exception as e:
            logging.error(f"Ошибка WebSocket потоков: {e}")

    async def handle_websocket_data(self, symbol: str, data: dict):
        """Обработка данных из WebSocket"""
        try:
            # Обновление данных в реальном времени
            if data.get("ohlcv"):
                candle = data["ohlcv"]
                formatted_data = [
                    {
                        "timestamp": candle[0],
                        "symbol": symbol,
                        "timeframe": "1m",
                        "open": candle[1],
                        "high": candle[2],
                        "low": candle[3],
                        "close": candle[4],
                        "volume": candle[5],
                    }
                ]

                await self.db_manager.insert_ohlcv_batch(formatted_data)

        except Exception as e:
            logging.error(f"Ошибка обработки WebSocket данных {symbol}: {e}")

    async def run(self):
        """Запуск основного цикла бота"""
        self.running = True

        tasks = []

        # Запуск сбора данных
        tasks.append(asyncio.create_task(self.start_data_collection()))

        # Запуск генерации сигналов
        tasks.append(asyncio.create_task(self.start_signal_generation()))

        # Запуск WebSocket потоков
        tasks.append(asyncio.create_task(self.start_websocket_streams()))

        # Запуск Telegram бота
        if self.telegram_bot:
            tasks.append(asyncio.create_task(self.telegram_bot.start_bot()))

        logging.info("Торговый бот запущен")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logging.info("Получен сигнал остановки")
        except Exception as e:
            logging.error(f"Критическая ошибка: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Корректное завершение работы бота"""
        logging.info("Завершение работы бота...")
        self.running = False

        if self.exchange:
            await self.exchange.close()

        if self.telegram_bot:
            await self.telegram_bot.stop_bot()

        logging.info("Бот остановлен")


def signal_handler(signum, frame):
    """Обработчик сигналов системы"""
    logging.info("Получен сигнал завершения")
    sys.exit(0)


async def main():
    """Главная функция"""
    # Настройка логирования
    setup_logging()

    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Создание и запуск бота
    bot = TradingBotS()

    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logging.error(f"Критическая ошибка запуска: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
