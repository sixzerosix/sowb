import logging
from typing import Dict, Optional
from datetime import datetime, timedelta


class RiskManager:
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
        self.daily_pnl = 0
        self.active_positions = {}

    async def calculate_position_size(
        self, symbol: str, signal_price: float, stop_loss: float, account_balance: float
    ) -> float:
        """Расчет размера позиции на основе риска"""
        try:
            # Риск на сделку в долларах
            risk_amount = account_balance * self.config.RISK_PER_TRADE

            # Расстояние до стоп-лосса
            price_risk = abs(signal_price - stop_loss)

            # Размер позиции
            position_size = risk_amount / price_risk

            # Ограничение максимального размера позиции
            max_position_value = (
                account_balance * 0.2
            )  # Максимум 20% баланса на позицию
            max_position_size = max_position_value / signal_price

            final_size = min(position_size, max_position_size)

            logging.info(
                f"Расчет позиции {symbol}: размер={final_size:.6f}, риск=${risk_amount:.2f}"
            )

            return final_size

        except Exception as e:
            logging.error(f"Ошибка расчета размера позиции: {e}")
            return 0

    async def can_open_position(self, symbol: str) -> bool:
        """Проверка возможности открытия позиции"""
        try:
            # Проверка максимального количества позиций
            if len(self.active_positions) >= self.config.MAX_POSITIONS:
                logging.warning(
                    f"Достигнут лимит позиций: {len(self.active_positions)}"
                )
                return False

            # Проверка существующей позиции по символу
            if symbol in self.active_positions:
                logging.warning(f"Позиция по {symbol} уже открыта")
                return False

            # Проверка дневного лимита убытков
            daily_stats = await self.get_daily_pnl()
            if daily_stats["total_pnl"] < -abs(
                daily_stats["account_balance"] * 0.05
            ):  # -5% дневной лимит
                logging.warning(
                    f"Достигнут дневной лимит убытков: {daily_stats['total_pnl']}"
                )
                return False

            return True

        except Exception as e:
            logging.error(f"Ошибка проверки возможности открытия позиции: {e}")
            return False

    async def get_daily_pnl(self) -> Dict:
        """Получение дневной статистики P&L"""
        try:
            stats = await self.db.get_strategy_performance(
                mode=self.config.TRADING_MODE, days=1
            )

            return {
                "total_pnl": stats.get("trades", {}).get("total_pnl", 0),
                "total_trades": stats.get("trades", {}).get("total_trades", 0),
                "win_rate": stats.get("trades", {}).get("win_rate", 0),
                "account_balance": 10000,  # Получить из биржи
            }

        except Exception as e:
            logging.error(f"Ошибка получения дневной статистики: {e}")
            return {
                "total_pnl": 0,
                "total_trades": 0,
                "win_rate": 0,
                "account_balance": 10000,
            }

    def add_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Добавление позиции в отслеживание"""
        self.active_positions[symbol] = {
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "timestamp": datetime.now(),
        }

    def remove_position(self, symbol: str):
        """Удаление позиции из отслеживания"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]

    async def update_daily_pnl(self, pnl: float):
        """Обновление дневного P&L"""
        self.daily_pnl += pnl
