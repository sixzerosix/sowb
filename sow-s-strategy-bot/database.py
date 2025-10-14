import aiosqlite
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import json
import logging


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_database(self):
        """Инициализация базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            # Таблица для OHLCV данных
            await db.execute(
                """
				CREATE TABLE IF NOT EXISTS ohlcv_data (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					timestamp DATETIME NOT NULL,
					symbol VARCHAR(20) NOT NULL,
					timeframe VARCHAR(10) NOT NULL,
					open REAL NOT NULL,
					high REAL NOT NULL,
					low REAL NOT NULL,
					close REAL NOT NULL,
					volume REAL NOT NULL,
					created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
					UNIQUE(timestamp, symbol, timeframe)
				);
			"""
            )

            # Таблица для торговых сигналов
            await db.execute(
                """
				CREATE TABLE IF NOT EXISTS trading_signals (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					timestamp DATETIME NOT NULL,
					symbol VARCHAR(20) NOT NULL,
					action VARCHAR(10) NOT NULL,
					price REAL NOT NULL,
					confidence REAL NOT NULL,
					ema_9 REAL,
					ema_21 REAL,
					rsi_14 REAL,
					adx REAL,
					bb_position REAL,
					vwap_deviation REAL,
					volume_ratio REAL,
					stop_loss REAL,
					take_profit REAL,
					risk_reward REAL,
					mode VARCHAR(10) NOT NULL,
					indicators TEXT,
					created_at DATETIME DEFAULT CURRENT_TIMESTAMP
				);
			"""
            )

            # Таблица для сделок
            await db.execute(
                """
				CREATE TABLE IF NOT EXISTS trades (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					timestamp DATETIME NOT NULL,
					symbol VARCHAR(20) NOT NULL,
					side VARCHAR(10) NOT NULL,
					amount REAL NOT NULL,
					price REAL NOT NULL,
					fee REAL DEFAULT 0,
					order_id VARCHAR(100),
					signal_id INTEGER,
					entry_price REAL,
					exit_price REAL,
					pnl REAL DEFAULT 0,
					pnl_pct REAL DEFAULT 0,
					status VARCHAR(20) DEFAULT 'open',
					mode VARCHAR(10) NOT NULL,
					created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
					closed_at DATETIME,
					FOREIGN KEY (signal_id) REFERENCES trading_signals (id)
				);
			"""
            )

            # Таблица для статистики
            await db.execute(
                """
				CREATE TABLE IF NOT EXISTS strategy_stats (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					date DATE NOT NULL,
					symbol VARCHAR(20) NOT NULL,
					mode VARCHAR(10) NOT NULL,
					total_signals INTEGER DEFAULT 0,
					buy_signals INTEGER DEFAULT 0,
					sell_signals INTEGER DEFAULT 0,
					executed_trades INTEGER DEFAULT 0,
					winning_trades INTEGER DEFAULT 0,
					losing_trades INTEGER DEFAULT 0,
					total_pnl REAL DEFAULT 0,
					win_rate REAL DEFAULT 0,
					avg_win REAL DEFAULT 0,
					avg_loss REAL DEFAULT 0,
					max_drawdown REAL DEFAULT 0,
					created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
					UNIQUE(date, symbol, mode)
				);
			"""
            )

            # Индексы для оптимизации
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data (symbol, timestamp DESC);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON trading_signals (symbol, timestamp DESC);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, timestamp DESC);"
            )

            await db.commit()

    async def insert_ohlcv_batch(self, data_list: List[Dict]):
        """Массовая вставка OHLCV данных"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
				INSERT OR REPLACE INTO ohlcv_data 
				(timestamp, symbol, timeframe, open, high, low, close, volume)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			""",
                [
                    (
                        datetime.fromtimestamp(d["timestamp"] / 1000),
                        d["symbol"],
                        d["timeframe"],
                        d["open"],
                        d["high"],
                        d["low"],
                        d["close"],
                        d["volume"],
                    )
                    for d in data_list
                ],
            )
            await db.commit()

    async def get_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> pd.DataFrame:
        """Получение OHLCV данных"""
        async with aiosqlite.connect(self.db_path) as db:
            query = """
				SELECT timestamp, open, high, low, close, volume
				FROM ohlcv_data
				WHERE symbol = ? AND timeframe = ?
				ORDER BY timestamp DESC
				LIMIT ?
			"""

            async with db.execute(query, (symbol, timeframe, limit)) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(
                rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

    async def save_signal(self, signal_data: Dict):
        """Сохранение торгового сигнала"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
				INSERT INTO trading_signals (
					timestamp, symbol, action, price, confidence,
					ema_9, ema_21, rsi_14, adx, bb_position,
					vwap_deviation, volume_ratio, stop_loss, take_profit,
					risk_reward, mode, indicators
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
                (
                    signal_data["timestamp"],
                    signal_data["symbol"],
                    signal_data["action"],
                    signal_data["price"],
                    signal_data["confidence"],
                    signal_data.get("ema_9"),
                    signal_data.get("ema_21"),
                    signal_data.get("rsi_14"),
                    signal_data.get("adx"),
                    signal_data.get("bb_position"),
                    signal_data.get("vwap_deviation"),
                    signal_data.get("volume_ratio"),
                    signal_data.get("stop_loss"),
                    signal_data.get("take_profit"),
                    signal_data.get("risk_reward"),
                    signal_data["mode"],
                    json.dumps(signal_data.get("indicators", {})),
                ),
            )
            await db.commit()

    async def get_strategy_performance(
        self, symbol: str = None, mode: str = None, days: int = 30
    ) -> Dict:
        """Получение статистики производительности стратегии"""
        async with aiosqlite.connect(self.db_path) as db:
            where_conditions = ["timestamp >= datetime('now', '-{} days')".format(days)]
            params = []

            if symbol:
                where_conditions.append("symbol = ?")
                params.append(symbol)
            if mode:
                where_conditions.append("mode = ?")
                params.append(mode)

            where_clause = " AND ".join(where_conditions)

            # Статистика сигналов
            signal_query = f"""
				SELECT 
					COUNT(*) as total_signals,
					SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
					SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
					AVG(confidence) as avg_confidence
				FROM trading_signals
				WHERE {where_clause}
			"""

            async with db.execute(signal_query, params) as cursor:
                signal_stats = await cursor.fetchone()

            # Статистика сделок
            trade_query = f"""
				SELECT 
					COUNT(*) as total_trades,
					SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
					SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
					SUM(pnl) as total_pnl,
					AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
					AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss,
					AVG(pnl_pct) as avg_return_pct
				FROM trades
				WHERE {where_clause} AND status = 'closed'
			"""

            async with db.execute(trade_query, params) as cursor:
                trade_stats = await cursor.fetchone()

            # Объединение статистики
            stats = {
                "signals": (
                    dict(zip([d[0] for d in cursor.description], signal_stats))
                    if signal_stats
                    else {}
                ),
                "trades": (
                    dict(zip([d[0] for d in cursor.description], trade_stats))
                    if trade_stats
                    else {}
                ),
            }

            # Расчет дополнительных метрик
            if stats["trades"].get("total_trades", 0) > 0:
                stats["trades"]["win_rate"] = (
                    stats["trades"]["winning_trades"] or 0
                ) / stats["trades"]["total_trades"]

                avg_win = stats["trades"]["avg_win"] or 0
                avg_loss = abs(stats["trades"]["avg_loss"] or 0)
                if avg_loss > 0:
                    stats["trades"]["profit_factor"] = avg_win / avg_loss
                else:
                    stats["trades"]["profit_factor"] = (
                        float("inf") if avg_win > 0 else 0
                    )

            return stats
