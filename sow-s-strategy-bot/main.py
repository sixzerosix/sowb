import asyncio
import logging
from datetime import datetime
import signal
import sys

from config import config
from database import DatabaseManager
from strategy import ScalpingStrategyS  # Скальпинг стратегия
from exchange import BybitPublicExchange
from telegram_bot import TelegramBot
from risk_manager import RiskManager
from utils import setup_logging


class HighFrequencyTradingBot:
    def __init__(self):
        self.config = config
        self.db_manager = None
        self.strategy = None
        self.exchange = None
        self.telegram_bot = None
        self.risk_manager = None
        self.running = False
        self.position_timers = {}  # Таймеры для автозакрытия позиций

    async def initialize(self):
        """Инициализация всех компонентов бота"""
        try:
            logging.info("🚀 Инициализация СКАЛЬПИНГ бота 'Стратегия S'")

            # Инициализация базы данных
            self.db_manager = DatabaseManager(self.config.DATABASE_PATH)
            await self.db_manager.init_database()
            logging.info("✅ База данных инициализирована")

            # Инициализация биржи
            self.exchange = BybitPublicExchange(self.config)
            await self.exchange.initialize()
            logging.info("✅ Подключение к Bybit установлено")

            # Инициализация скальпинг стратегии
            self.strategy = ScalpingStrategyS(self.config, self.db_manager)
            logging.info("✅ Скальпинг стратегия инициализирована")

            # Инициализация риск-менеджера
            self.risk_manager = RiskManager(self.config, self.db_manager)
            logging.info("✅ Риск-менеджер инициализирован")

            # Инициализация Telegram бота
            if self.config.TELEGRAM_TOKEN:
                self.telegram_bot = TelegramBot(
                    self.config, self.db_manager, self.strategy, self.exchange
                )
                logging.info("✅ Telegram бот инициализирован")

            logging.info("🎯 ВСЕ КОМПОНЕНТЫ СКАЛЬПИНГ БОТА ГОТОВЫ!")
            logging.info(f"⚡ Режим: {self.config.TRADING_MODE}")
            logging.info(f"⚡ Таймфрейм: {self.config.TIMEFRAME}")
            logging.info(f"⚡ Плечо: {self.config.LEVERAGE}x")
            logging.info(f"⚡ Символы: {self.config.SYMBOLS}")

        except Exception as e:
            logging.error(f"❌ Ошибка инициализации: {e}")
            raise

    async def start_data_collection(self):
        """БЫСТРЫЙ сбор данных для скальпинга"""
        while self.running:
            try:
                # Получение 1-минутных данных (быстро!)
                ohlcv_data = await self.exchange.fetch_ohlcv_batch(
                    self.config.SYMBOLS,
                    self.config.TIMEFRAME,  # 1m
                    50,  # Меньше данных = больше скорость
                )

                # Быстрое сохранение только последних свечей
                for symbol, data in ohlcv_data.items():
                    if data:
                        formatted_data = []
                        # Берем только последние 10 свечей для скорости
                        for candle in data[-10:]:
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

                logging.info(
                    f"⚡ СКАЛЬПИНГ: Данные обновлены для {len(ohlcv_data)} символов"
                )

                # БЫСТРАЯ пауза для скальпинга
                await asyncio.sleep(
                    getattr(self.config, "DATA_UPDATE_INTERVAL", 10)
                )  # 10 секунд

            except Exception as e:
                logging.error(f"❌ Ошибка скальпинг сбора данных: {e}")
                await asyncio.sleep(30)

    async def start_signal_generation(self):
        """БЫСТРАЯ генерация сигналов для скальпинга"""
        while self.running:
            try:
                for symbol in self.config.SYMBOLS:
                    # Скальпинг анализ (быстрый!)
                    signal = await self.strategy.analyze_symbol_scalping(symbol)

                    if signal.action != "HOLD":
                        logging.info(
                            f"⚡ СКАЛЬПИНГ СИГНАЛ {symbol}: {signal.action} at ${signal.price:.4f} "
                            f"(confidence: {signal.confidence:.2%}) "
                            f"RR: 1:{signal.risk_reward:.2f} "
                            f"SL: ${signal.stop_loss:.2f} TP: ${signal.take_profit:.2f}"
                        )

                        # Telegram уведомление
                        if self.telegram_bot:
                            await self.send_scalping_notification(signal)

                        # Быстрое выполнение при достаточной уверенности
                        min_confidence = getattr(
                            self.config, "MIN_CONFIDENCE_FOR_TRADE", 0.6
                        )
                        if signal.confidence > min_confidence:
                            await self.execute_scalping_signal(signal)

                    # Быстрый переход к следующему символу
                    await asyncio.sleep(0.5)  # 0.5 секунды между символами

                # БЫСТРАЯ пауза между циклами
                signal_interval = getattr(self.config, "SIGNAL_CHECK_INTERVAL", 5)
                await asyncio.sleep(signal_interval)  # 5 секунд

            except Exception as e:
                logging.error(f"❌ Ошибка генерации скальпинг сигналов: {e}")
                await asyncio.sleep(10)

    async def execute_scalping_signal(self, signal):
        """Выполнение скальпинг сигнала с плечом"""
        try:
            # Проверка возможности открытия позиции
            if not await self.risk_manager.can_open_position(signal.symbol):
                logging.warning(
                    f"⚠️ СКАЛЬПИНГ позицию по {signal.symbol} открыть нельзя"
                )
                return

            # Получение баланса
            account_info = await self.exchange.get_account_info()
            balance = account_info.get("balance", {}).get("USDT", {}).get("free", 0)

            min_balance = getattr(self.config, "MIN_BALANCE_FOR_SCALPING", 50)
            if balance < min_balance:
                logging.warning(f"⚠️ Недостаточный баланс для скальпинга: ${balance}")
                return

            # Расчет размера позиции
            base_position_size = await self.risk_manager.calculate_position_size(
                signal.symbol, signal.price, signal.stop_loss, balance
            )

            # Применение плеча для скальпинга
            leverage = getattr(self.config, "LEVERAGE", 1)
            min_confidence_for_leverage = getattr(
                self.config, "MIN_CONFIDENCE_FOR_LEVERAGE", 0.8
            )

            if signal.confidence > min_confidence_for_leverage:
                position_size = base_position_size * leverage
                actual_leverage = leverage
            else:
                position_size = base_position_size
                actual_leverage = 1

            if position_size <= 0:
                return

            # Размещение ДЕМО ордера для скальпинга
            order = await self.place_scalping_order(
                signal, position_size, actual_leverage
            )

            if order and not order.get("error"):
                logging.info(
                    f"🎯 СКАЛЬПИНГ ОРДЕР ИСПОЛНЕН: {signal.symbol} {signal.action} "
                    f"{position_size:.6f} at ${signal.price:.4f} "
                    f"Плечо: {actual_leverage}x "
                    f"Confidence: {signal.confidence:.1%}"
                )

                # Запуск таймера автозакрытия для скальпинга
                max_duration = getattr(
                    self.config, "MAX_TRADE_DURATION", 900
                )  # 15 минут
                asyncio.create_task(
                    self.auto_close_scalping_position(signal.symbol, max_duration)
                )

            else:
                error_msg = (
                    order.get("error", "Неизвестная ошибка")
                    if order
                    else "Ордер не создан"
                )
                logging.error(
                    f"❌ Ошибка скальпинг ордера {signal.symbol}: {error_msg}"
                )

        except Exception as e:
            logging.error(
                f"❌ Ошибка выполнения скальпинг сигнала {signal.symbol}: {e}"
            )

    async def place_scalping_order(self, signal, position_size, leverage):
        """Размещение скальпинг ордера"""
        try:
            order = await self.exchange.place_order(
                symbol=signal.symbol,
                side=signal.action.lower(),
                amount=position_size,
                price=None,  # Market order для скорости
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            # Добавляем информацию о плече в ордер
            if order and not order.get("error"):
                order["leverage"] = leverage
                order["scalping_mode"] = True

            return order

        except Exception as e:
            logging.error(f"❌ Ошибка размещения скальпинг ордера: {e}")
            return {"error": str(e)}

    async def auto_close_scalping_position(self, symbol: str, max_duration: int):
        """Автоматическое закрытие скальпинг позиции по времени"""
        try:
            await asyncio.sleep(max_duration)

            # Проверяем, есть ли еще позиция
            if (
                hasattr(self.exchange, "demo_positions")
                and symbol in self.exchange.demo_positions
            ):
                logging.info(
                    f"⏰ АВТОЗАКРЫТИЕ скальпинг позиции {symbol} (макс. время: {max_duration//60} мин)"
                )

                # Закрываем позицию
                result = await self.exchange.close_demo_position(symbol)

                if result and not result.get("error"):
                    pnl = result.get("pnl", 0)
                    logging.info(
                        f"✅ Позиция {symbol} закрыта автоматически. P&L: ${pnl:.2f}"
                    )

                    # Уведомление в Telegram
                    if self.telegram_bot:
                        await self.send_position_closed_notification(
                            symbol, pnl, "Автозакрытие по времени"
                        )

        except Exception as e:
            logging.error(f"❌ Ошибка автозакрытия позиции {symbol}: {e}")

    async def send_scalping_notification(self, signal):
        """Отправка уведомления о скальпинг сигнале"""
        try:
            if not self.telegram_bot:
                return

            emoji = "🟢" if signal.action == "BUY" else "🔴"
            leverage = getattr(self.config, "LEVERAGE", 1)

            message = f"{emoji} <b>⚡ СКАЛЬПИНГ СИГНАЛ</b>\n\n"
            message += f"<b>Символ:</b> {signal.symbol}\n"
            message += f"<b>Действие:</b> {signal.action}\n"
            message += f"<b>Цена:</b> ${signal.price:.4f}\n"
            message += f"<b>Уверенность:</b> {signal.confidence:.1%}\n"
            message += f"<b>Плечо:</b> {leverage}x\n"

            if signal.stop_loss:
                message += f"<b>Stop Loss:</b> ${signal.stop_loss:.4f}\n"
            if signal.take_profit:
                message += f"<b>Take Profit:</b> ${signal.take_profit:.4f}\n"
            if signal.risk_reward:
                message += f"<b>Risk/Reward:</b> 1:{signal.risk_reward:.2f}\n"

            message += f"\n<b>⚡ Скальпинг режим</b>\n"
            message += f"<i>Время: {signal.timestamp.strftime('%H:%M:%S')}</i>"

            await self.telegram_bot.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID, text=message, parse_mode="HTML"
            )

        except Exception as e:
            logging.error(f"❌ Ошибка отправки скальпинг уведомления: {e}")

    async def send_position_closed_notification(
        self, symbol: str, pnl: float, reason: str
    ):
        """Уведомление о закрытии позиции"""
        try:
            if not self.telegram_bot:
                return

            emoji = "✅" if pnl >= 0 else "❌"

            message = f"{emoji} <b>Позиция закрыта</b>\n\n"
            message += f"<b>Символ:</b> {symbol}\n"
            message += f"<b>P&L:</b> ${pnl:.2f}\n"
            message += f"<b>Причина:</b> {reason}\n"
            message += f"<i>Время: {datetime.now().strftime('%H:%M:%S')}</i>"

            await self.telegram_bot.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID, text=message, parse_mode="HTML"
            )

        except Exception as e:
            logging.error(f"❌ Ошибка уведомления о закрытии: {e}")

    async def start_websocket_streams(self):
        """Запуск быстрых потоков данных"""
        try:
            await self.exchange.start_websocket_streams(
                self.config.SYMBOLS, self.handle_websocket_data
            )
        except Exception as e:
            logging.error(f"❌ Ошибка WebSocket потоков: {e}")

    async def handle_websocket_data(self, symbol: str, data: dict):
        """Обработка данных из WebSocket"""
        try:
            # Быстрая обработка для скальпинга
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
            logging.error(f"❌ Ошибка обработки WebSocket данных {symbol}: {e}")

    async def run(self):
        """Запуск основного цикла скальпинг бота"""
        self.running = True

        tasks = []

        # БЫСТРЫЙ сбор данных
        tasks.append(asyncio.create_task(self.start_data_collection()))

        # БЫСТРАЯ генерация сигналов
        tasks.append(asyncio.create_task(self.start_signal_generation()))

        # WebSocket потоки (опционально)
        tasks.append(asyncio.create_task(self.start_websocket_streams()))

        # Telegram бот
        if self.telegram_bot:
            tasks.append(asyncio.create_task(self.telegram_bot.start_bot()))

        logging.info("🚀⚡ СКАЛЬПИНГ БОТ ЗАПУЩЕН И ГОТОВ К ТОРГОВЛЕ!")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logging.info("⛔ Получен сигнал остановки")
        except Exception as e:
            logging.error(f"💥 Критическая ошибка: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Корректное завершение работы скальпинг бота"""
        logging.info("🛑 Завершение работы скальпинг бота...")
        self.running = False

        # Закрываем все открытые позиции
        if hasattr(self.exchange, "demo_positions"):
            for symbol in list(self.exchange.demo_positions.keys()):
                try:
                    await self.exchange.close_demo_position(symbol)
                    logging.info(f"✅ Позиция {symbol} закрыта при завершении")
                except Exception as e:
                    logging.error(f"❌ Ошибка закрытия позиции {symbol}: {e}")

        if self.exchange:
            self.exchange.close()

        if self.telegram_bot:
            await self.telegram_bot.stop_bot()

        logging.info("✅ Скальпинг бот остановлен")


def signal_handler(signum, frame):
    """Обработчик сигналов системы"""
    logging.info("⛔ Получен сигнал завершения")
    sys.exit(0)


async def main():
    """Главная функция"""
    # Настройка логирования
    setup_logging()

    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Создание и запуск скальпинг бота
    bot = HighFrequencyTradingBot()

    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logging.error(f"💥 Критическая ошибка запуска: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
