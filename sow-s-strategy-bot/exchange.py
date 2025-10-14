import ccxt
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


class BybitPublicExchange:
    def __init__(self, config):
        self.config = config
        self.exchange = None
        self.is_connected = False
        self.demo_balance = 10000
        self.demo_positions = {}

    async def initialize(self):
        """Инициализация публичного подключения к Bybit"""
        try:
            # Синхронное подключение БЕЗ API ключей
            self.exchange = ccxt.bybit(
                {
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "linear",  # USDT perpetual
                    },
                }
            )

            # Загружаем рынки
            markets = self.exchange.load_markets()

            # ОТЛАДКА: Выводим доступные символы
            logging.info(f"Всего рынков: {len(markets)}")

            # Ищем BTC символы
            btc_symbols = [
                symbol
                for symbol in markets.keys()
                if "BTC" in symbol and "USDT" in symbol
            ]
            logging.info(f"BTC символы: {btc_symbols[:10]}")  # Первые 10

            # Ищем ETH символы
            eth_symbols = [
                symbol
                for symbol in markets.keys()
                if "ETH" in symbol and "USDT" in symbol
            ]
            logging.info(f"ETH символы: {eth_symbols[:5]}")  # Первые 5

            # Пробуем разные варианты BTC символа
            possible_btc_symbols = ["BTCUSDT", "BTC/USDT", "BTCUSD", "BTC/USD"]
            btc_symbol = None

            for symbol in possible_btc_symbols:
                if symbol in markets:
                    btc_symbol = symbol
                    break

            if btc_symbol:
                # Тестируем получение тикера
                ticker = self.exchange.fetch_ticker(btc_symbol)

                self.is_connected = True
                logging.info(f"Подключение к Bybit PUBLIC API успешно")
                logging.info(f"Найден BTC символ: {btc_symbol}")
                logging.info(f"Тестовый баланс USDT: {self.demo_balance}")
                logging.info(f"BTC цена: ${ticker['last']}")

                # Обновляем конфигурацию с правильными символами
                self._update_symbols_config(markets)

            else:
                # Если BTC не найден, выводим все доступные символы
                all_symbols = list(markets.keys())[:20]  # Первые 20
                logging.error(f"BTC символ не найден. Доступные символы: {all_symbols}")
                raise Exception("BTC символ не найден в списке рынков")

        except Exception as e:
            logging.error(f"Ошибка подключения к Bybit PUBLIC API: {e}")
            self.is_connected = False
            raise

    def _update_symbols_config(self, markets):
        """Обновление символов в конфигурации: проверяет, существуют ли символы из config.SYMBOLS на бирже."""
        logging.info(
            f"Начало проверки символов. Исходные символы из config: {self.config.SYMBOLS}"
        )

        # Для отладки: выводим первые 50 ключей из markets
        logging.debug(f"Первые 50 ключей из markets: {list(markets.keys())[:50]}")

        valid_symbols = []
        for original_symbol in self.config.SYMBOLS:
            if original_symbol in markets:
                valid_symbols.append(original_symbol)
                logging.info(f"Символ '{original_symbol}' успешно найден на Bybit.")
            else:
                logging.warning(
                    f"Символ '{original_symbol}' из конфигурации не найден на Bybit. Он будет пропущен."
                )

        self.config.SYMBOLS = valid_symbols
        logging.info(f"Финальный список символов для торговли: {self.config.SYMBOLS}")

    async def fetch_ohlcv_batch(
        self, symbols: List[str], timeframe: str = "1h", limit: int = 100
    ) -> Dict:
        """Получение РЕАЛЬНЫХ OHLCV данных через публичный API"""
        results = {}

        for symbol in symbols:
            try:
                # Синхронный вызов в async функции
                ohlcv = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda s=symbol: self.exchange.fetch_ohlcv(s, timeframe, limit),
                )
                results[symbol] = ohlcv

                logging.info(f"Получено {len(ohlcv)} свечей для {symbol}")
                await asyncio.sleep(0.2)  # Rate limiting

            except Exception as e:
                logging.error(f"Ошибка получения данных {symbol}: {e}")
                results[symbol] = []

        return results

    async def get_real_time_ticker(self, symbol: str) -> Dict:
        """Получение тикера в реальном времени"""
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_ticker(symbol)
            )
            return {
                "symbol": symbol,
                "price": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "volume": ticker["baseVolume"],
                "change": ticker["change"],
                "percentage": ticker["percentage"],
                "timestamp": ticker["timestamp"],
            }
        except Exception as e:
            logging.error(f"Ошибка получения тикера {symbol}: {e}")
            return {}

    async def get_account_info(self) -> Dict:
        """ДЕМО информация об аккаунте"""
        total_pnl = 0
        for symbol, position in self.demo_positions.items():
            try:
                current_ticker = await self.get_real_time_ticker(symbol)
                current_price = current_ticker.get("price", position["entry_price"])

                if position["side"] == "buy":
                    pnl = (current_price - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - current_price) * position["size"]

                position["unrealized_pnl"] = pnl
                total_pnl += pnl

            except Exception as e:
                logging.error(f"Ошибка расчета P&L для {symbol}: {e}")

        return {
            "balance": {
                "USDT": {
                    "free": self.demo_balance + total_pnl,
                    "used": sum(
                        pos.get("margin", 0) for pos in self.demo_positions.values()
                    ),
                    "total": self.demo_balance + total_pnl,
                }
            },
            "active_positions": list(self.demo_positions.values()),
            "total_equity": self.demo_balance + total_pnl,
            "unrealized_pnl": total_pnl,
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> Dict:
        """Размещение ДЕМО ордера"""
        try:
            if price is None:
                ticker = await self.get_real_time_ticker(symbol)
                price = ticker.get("price")

            if not price:
                return {"error": "Не удалось получить цену"}

            order_id = f"demo_{int(datetime.now().timestamp())}"
            margin_required = amount * price * 0.01

            available_balance = self.demo_balance + sum(
                pos.get("unrealized_pnl", 0) for pos in self.demo_positions.values()
            )
            if margin_required > available_balance:
                return {"error": "Недостаточно средств"}

            demo_order = {
                "id": order_id,
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "filled",
                "timestamp": datetime.now(),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "margin": margin_required,
                "demo_mode": True,
            }

            self.demo_positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "size": amount,
                "entry_price": price,
                "margin": margin_required,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timestamp": datetime.now(),
                "unrealized_pnl": 0,
            }

            self.demo_balance -= margin_required

            logging.info(
                f"ДЕМО ордер исполнен: {symbol} {side} {amount:.6f} at ${price:.4f}"
            )

            return demo_order

        except Exception as e:
            logging.error(f"Ошибка размещения демо ордера {symbol} {side}: {e}")
            return {"error": str(e)}

    async def start_websocket_streams(self, symbols: List[str], callback):
        """Запуск потока реальных данных"""
        try:
            for symbol in symbols:
                asyncio.create_task(self._real_time_stream(symbol, callback))
                await asyncio.sleep(0.1)

            logging.info(f"Потоки реальных данных запущены для {len(symbols)} символов")

        except Exception as e:
            logging.error(f"Ошибка запуска потоков данных: {e}")

    async def _real_time_stream(self, symbol: str, callback):
        """Поток реальных данных для одного символа"""
        while True:
            try:
                ticker = await self.get_real_time_ticker(symbol)

                ohlcv = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.exchange.fetch_ohlcv(symbol, "1m", 1)
                )

                await callback(
                    symbol,
                    {
                        "ticker": ticker,
                        "ohlcv": ohlcv[0] if ohlcv else None,
                        "timestamp": datetime.now(),
                    },
                )

                await asyncio.sleep(30)

            except Exception as e:
                logging.error(f"Ошибка потока данных {symbol}: {e}")
                await asyncio.sleep(60)

    def close(self):
        """Закрытие подключения"""
        if self.exchange:
            self.is_connected = False
            logging.info("Подключение к Bybit PUBLIC API закрыто")
