import asyncio
from email.message import Message
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from datetime import datetime, timedelta
import pandas as pd


class TelegramBot:
    def __init__(self, config, db_manager, strategy, exchange):
        self.config = config
        self.db = db_manager
        self.strategy = strategy
        self.exchange = exchange
        self.bot = Bot(token=config.TELEGRAM_TOKEN)
        self.dp = Dispatcher()
        self.setup_handlers()

    def setup_handlers(self):
        """Настройка обработчиков команд"""

        @self.dp.message(Command("start"))
        async def start_command(message: types.Message):
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="📊 Статистика", callback_data="stats")],
                    [InlineKeyboardButton(text="💰 Баланс", callback_data="balance")],
                    [
                        InlineKeyboardButton(
                            text="📈 Позиции", callback_data="positions"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            text="⚙️ Настройки", callback_data="settings"
                        )
                    ],
                    [InlineKeyboardButton(text="🔄 Статус", callback_data="status")],
                ]
            )

            await message.answer(
                "🤖 <b>Торговый бот 'Стратегия S'</b>\n\n"
                f"Режим: <code>{self.config.TRADING_MODE}</code>\n"
                f"Символы: <code>{', '.join(self.config.SYMBOLS)}</code>\n"
                f"Таймфрейм: <code>{self.config.TIMEFRAME}</code>\n\n"
                "Выберите действие:",
                parse_mode="HTML",
                reply_markup=keyboard,
            )

        @self.dp.callback_query(lambda c: c.data == "stats")
        async def stats_callback(callback: types.CallbackQuery):
            await self.send_statistics(callback.message)
            await callback.answer()

        @self.dp.callback_query(lambda c: c.data == "balance")
        async def balance_callback(callback: types.CallbackQuery):
            await self.send_balance_info(callback.message)
            await callback.answer()

        @self.dp.callback_query(lambda c: c.data == "positions")
        async def positions_callback(callback: types.CallbackQuery):
            await self.send_positions_info(callback.message)
            await callback.answer()

        @self.dp.callback_query(lambda c: c.data == "settings")
        async def settings_callback(callback: types.CallbackQuery):
            await self.send_settings_info(callback.message)
            await callback.answer()

        @self.dp.callback_query(lambda c: c.data == "status")
        async def status_callback(callback: types.CallbackQuery):
            await self.send_status_info(callback.message)
            await callback.answer()

        @self.dp.message(Command("stats"))
        async def stats_command(message: types.Message):
            await self.send_statistics(message)

        @self.dp.message(Command("balance"))
        async def balance_command(message: types.Message):
            await self.send_balance_info(message)

        @self.dp.message(Command("positions"))
        async def positions_command(message: types.Message):
            await self.send_positions_info(message)

        @self.dp.message(Command("stop"))
        async def stop_command(message: types.Message):
            await message.answer("⛔ Бот остановлен администратором")
            # Здесь можно добавить логику остановки бота

        @self.dp.message(Command("myid"))
        async def get_my_chat_id_handler(message: types.Message) -> None:
            """
            Обработчик команды /myid. Отправляет пользователю его chat_id.
            """
            if message.chat:
                await message.answer(f"Ваш Chat ID: `{message.chat.id}`")
            else:
                await message.answer("Не удалось определить ваш Chat ID.")

    async def send_statistics(self, message):
        """Отправка статистики производительности"""
        try:
            stats = await self.db.get_strategy_performance(
                mode=self.config.TRADING_MODE, days=7
            )

            signals_stats = stats.get("signals", {})
            trades_stats = stats.get("trades", {})

            text = f"📊 <b>Статистика за 7 дней</b>\n\n"
            text += f"<b>Сигналы:</b>\n"
            text += f"• Всего: {signals_stats.get('total_signals', 0)}\n"
            text += f"• BUY: {signals_stats.get('buy_signals', 0)}\n"
            text += f"• SELL: {signals_stats.get('sell_signals', 0)}\n"
            text += (
                f"• Ср. уверенность: {float(signals_stats.get('avg_confidence', 0)):.2%}\n\n"
            )

            text += f"<b>Сделки:</b>\n"
            text += f"• Всего: {trades_stats.get('total_trades', 0)}\n"
            text += f"• Прибыльных: {trades_stats.get('winning_trades', 0)}\n"
            text += f"• Убыточных: {trades_stats.get('losing_trades', 0)}\n"
            text += f"• Винрейт: {float(trades_stats.get('win_rate', 0)):.2%}\n"
            text += f"• Общий P&L: ${trades_stats.get('total_pnl', 0):.2f}\n"
            text += f"• Средняя прибыль: ${trades_stats.get('avg_win', 0):.2f}\n"
            text += f"• Средний убыток: ${trades_stats.get('avg_loss', 0):.2f}\n"

            if trades_stats.get("profit_factor"):
                text += f"• Profit Factor: {trades_stats.get('profit_factor', 0):.2f}\n"

            await message.answer(text, parse_mode="HTML")

        except Exception as e:
            await message.answer(f"❌ Ошибка получения статистики: {e}")

    async def send_balance_info(self, message):
        """Отправка информации о балансе"""
        try:
            if self.exchange.is_connected:
                account_info = await self.exchange.get_account_info()
                balance = account_info.get("balance", {})

                text = f"💰 <b>Баланс аккаунта</b>\n\n"
                text += f"USDT: {balance.get('USDT', {}).get('free', 0):.2f}\n"
                text += f"Заблокировано: {balance.get('USDT', {}).get('used', 0):.2f}\n"
                text += (
                    f"Общий баланс: {balance.get('USDT', {}).get('total', 0):.2f}\n\n"
                )

                active_positions = account_info.get("active_positions", [])
                if active_positions:
                    text += f"<b>Активные позиции:</b>\n"
                    for pos in active_positions:
                        text += (
                            f"• {pos['symbol']}: {pos['side']} {pos['contracts']:.4f}\n"
                        )
                else:
                    text += "Активных позиций нет"

            else:
                text = "❌ Нет подключения к бирже"

            await message.answer(text, parse_mode="HTML")

        except Exception as e:
            await message.answer(f"❌ Ошибка получения баланса: {e}")

    async def send_positions_info(self, message):
        """Отправка информации о позициях"""
        try:
            # Получение активных сделок из БД
            async with self.db.db_path as db:
                async with db.execute(
                    """
					SELECT symbol, side, amount, entry_price, pnl, created_at
					FROM trades 
					WHERE status = 'open' AND mode = ?
					ORDER BY created_at DESC
				""",
                    (self.config.TRADING_MODE,),
                ) as cursor:
                    trades = await cursor.fetchall()

            if trades:
                text = f"📈 <b>Открытые позиции</b>\n\n"
                for trade in trades:
                    symbol, side, amount, entry_price, pnl, created_at = trade
                    text += f"<b>{symbol}</b>\n"
                    text += f"• Направление: {side}\n"
                    text += f"• Размер: {amount:.6f}\n"
                    text += f"• Цена входа: ${entry_price:.4f}\n"
                    text += f"• P&L: ${pnl:.2f}\n"
                    text += f"• Время: {created_at}\n\n"
            else:
                text = "Открытых позиций нет"

            await message.answer(text, parse_mode="HTML")

        except Exception as e:
            await message.answer(f"❌ Ошибка получения позиций: {e}")

    async def send_settings_info(self, message):
        """Отправка текущих настроек"""
        text = f"⚙️ <b>Текущие настройки</b>\n\n"
        text += f"<b>Основные:</b>\n"
        text += f"• Режим: {self.config.TRADING_MODE}\n"
        text += f"• Символы: {', '.join(self.config.SYMBOLS)}\n"
        text += f"• Таймфрейм: {self.config.TIMEFRAME}\n\n"

        text += f"<b>Стратегия:</b>\n"
        text += f"• EMA быстрая: {self.config.EMA_FAST}\n"
        text += f"• EMA медленная: {self.config.EMA_SLOW}\n"
        text += f"• RSI период: {self.config.RSI_PERIOD_LONG}\n"
        text += f"• Мин. ADX: {self.config.MIN_ADX}\n\n"

        text += f"<b>Риск-менеджмент:</b>\n"
        text += f"• Риск на сделку: {self.config.RISK_PER_TRADE:.1%}\n"
        text += f"• ATR стоп: {self.config.ATR_STOP_MULTIPLIER}x\n"
        text += f"• ATR цель: {self.config.ATR_TARGET_MULTIPLIER}x\n"
        text += f"• Макс. позиций: {self.config.MAX_POSITIONS}\n"

        await message.answer(text, parse_mode="HTML")

    async def send_status_info(self, message):
        """Отправка статуса системы"""
        text = f"🔄 <b>Статус системы</b>\n\n"
        text += (
            f"• Подключение к Bybit: {'✅' if self.exchange.is_connected else '❌'}\n"
        )
        text += f"• База данных: ✅\n"
        text += f"• Режим торговли: {self.config.TRADING_MODE}\n"
        text += f"• Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        await message.answer(text, parse_mode="HTML")

    async def send_signal_notification(self, signal):
        """Отправка уведомления о торговом сигнале"""
        try:
            emoji = (
                "🟢"
                if signal.action == "BUY"
                else "🔴" if signal.action == "SELL" else "⚪"
            )

            text = f"{emoji} <b>Торговый сигнал</b>\n\n"
            text += f"<b>Символ:</b> {signal.symbol}\n"
            text += f"<b>Действие:</b> {signal.action}\n"
            text += f"<b>Цена:</b> ${signal.price:.4f}\n"
            text += f"<b>Уверенность:</b> {signal.confidence:.1%}\n"

            if signal.stop_loss:
                text += f"<b>Stop Loss:</b> ${signal.stop_loss:.4f}\n"
            if signal.take_profit:
                text += f"<b>Take Profit:</b> ${signal.take_profit:.4f}\n"
            if signal.risk_reward:
                text += f"<b>Risk/Reward:</b> 1:{signal.risk_reward:.2f}\n"

            text += f"\n<b>Индикаторы:</b>\n"
            text += f"• RSI: {signal.indicators.get('rsi_14', 0):.1f}\n"
            text += f"• ADX: {signal.indicators.get('adx', 0):.1f}\n"
            text += f"• Объем: {signal.indicators.get('volume_ratio', 0):.2f}x\n"

            text += f"\n<i>Время: {signal.timestamp.strftime('%H:%M:%S')}</i>"

            await self.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID, text=text, parse_mode="HTML"
            )

        except Exception as e:
            logging.error(f"Ошибка отправки уведомления в Telegram: {e}")

    async def start_bot(self):
        """Запуск Telegram бота"""
        try:
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logging.error(f"Ошибка запуска Telegram бота: {e}")

    async def stop_bot(self):
        """Остановка Telegram бота"""
        await self.bot.session.close()
