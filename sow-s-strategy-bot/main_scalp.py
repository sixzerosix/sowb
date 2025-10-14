# В начале файла
from config import config  # Теперь это HighFrequencyTradingConfig
from _strategy import ScalpingStrategyS  # Новая стратегия


class HighFrequencyTradingBot:
    def __init__(self):
        self.config = config
        # ... остальное как было

    async def start_data_collection(self):
        """БЫСТРЫЙ сбор данных для скальпинга"""
        while self.running:
            try:
                # Получение 1-минутных данных
                ohlcv_data = await self.exchange.fetch_ohlcv_batch(
                    self.config.SYMBOLS,
                    "1m",  # 1-минутные свечи
                    50,  # Меньше истории, больше скорость
                )

                # Быстрое сохранение
                for symbol, data in ohlcv_data.items():
                    if data:
                        formatted_data = []
                        for candle in data[-10:]:  # Только последние 10 свечей
                            formatted_data.append(
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
                            )

                        await self.db_manager.insert_ohlcv_batch(formatted_data)

                logging.info(
                    f"СКАЛЬПИНГ: Данные обновлены для {len(ohlcv_data)} символов"
                )

                # БЫСТРАЯ пауза
                await asyncio.sleep(self.config.DATA_UPDATE_INTERVAL)  # 10 секунд

            except Exception as e:
                logging.error(f"Ошибка скальпинг сбора данных: {e}")
                await asyncio.sleep(30)

    async def start_signal_generation(self):
        """БЫСТРАЯ генерация сигналов"""
        while self.running:
            try:
                for symbol in self.config.SYMBOLS:
                    # Скальпинг анализ
                    signal = await self.strategy.analyze_symbol_scalping(symbol)

                    if signal.action != "HOLD":
                        logging.info(
                            f"⚡ СКАЛЬПИНГ СИГНАЛ {symbol}: {signal.action} at ${signal.price:.4f} "
                            f"(confidence: {signal.confidence:.2%}) "
                            f"RR: 1:{signal.risk_reward:.2f}"
                        )

                        # Telegram уведомление
                        if self.telegram_bot:
                            await self.telegram_bot.send_scalping_signal_notification(
                                signal
                            )

                        # Быстрое выполнение при высокой уверенности
                        if signal.confidence > self.config.MIN_CONFIDENCE_FOR_TRADE:
                            await self.execute_scalping_signal(signal)

                    await asyncio.sleep(0.5)  # 0.5 секунды между символами

                # БЫСТРАЯ пауза между циклами
                await asyncio.sleep(self.config.SIGNAL_CHECK_INTERVAL)  # 5 секунд

            except Exception as e:
                logging.error(f"Ошибка генерации скальпинг сигналов: {e}")
                await asyncio.sleep(10)

    async def execute_scalping_signal(self, signal):
        """Выполнение скальпинг сигнала с плечом"""
        try:
            # Проверка возможности открытия позиции
            if not await self.risk_manager.can_open_position(signal.symbol):
                return

            # Получение баланса
            account_info = await self.exchange.get_account_info()
            balance = account_info.get("balance", {}).get("USDT", {}).get("free", 0)

            if balance < 50:  # Минимум для скальпинга
                return

            # Расчет размера позиции с плечом
            base_position_size = await self.risk_manager.calculate_position_size(
                signal.symbol, signal.price, signal.stop_loss, balance
            )

            # Применение плеча
            leverage_multiplier = (
                self.config.LEVERAGE
                if signal.confidence > self.config.MIN_CONFIDENCE_FOR_LEVERAGE
                else 1
            )
            position_size = base_position_size * leverage_multiplier

            if position_size <= 0:
                return

            # Размещение ДЕМО ордера с плечом
            order = await self.exchange.place_leveraged_demo_order(
                symbol=signal.symbol,
                side=signal.action.lower(),
                amount=position_size,
                price=signal.price,
                leverage=leverage_multiplier,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            if order and not order.get("error"):
                logging.info(
                    f"⚡ СКАЛЬПИНГ ОРДЕР: {signal.symbol} {signal.action} "
                    f"{position_size:.6f} at ${signal.price:.4f} "
                    f"Плечо: {leverage_multiplier}x"
                )

                # Установка таймера для принудительного закрытия
                asyncio.create_task(
                    self.auto_close_position(
                        signal.symbol, self.config.MAX_TRADE_DURATION
                    )
                )

        except Exception as e:
            logging.error(f"Ошибка выполнения скальпинг сигнала: {e}")

    async def auto_close_position(self, symbol: str, max_duration: int):
        """Автоматическое закрытие позиции через время"""
        await asyncio.sleep(max_duration)  # Ждем максимальное время

        # Проверяем, есть ли еще позиция
        if symbol in self.exchange.demo_positions:
            logging.info(f"⏰ ПРИНУДИТЕЛЬНОЕ ЗАКРЫТИЕ {symbol} (макс. время)")
            await self.exchange.close_demo_position(symbol)
