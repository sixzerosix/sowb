import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
from indicators import TechnicalIndicators
import logging


# ДОБАВИТЬ ЭТО! 👇
@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    confidence: float
    indicators: Dict
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None


class ScalpingStrategyS:
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
        self.indicators = TechnicalIndicators()
        self.last_signals = {}  # Кэш последних сигналов

    async def analyze_symbol_scalping(self, symbol: str) -> TradingSignal:
        """Скальпинг анализ символа"""
        try:
            # Получение 1-минутных данных
            df = await self.db.get_ohlcv(symbol, "1m", 200)  # 200 минут истории

            if df.empty or len(df) < 50:
                return self._create_hold_signal(symbol, 0, "Недостаточно данных")

            # Добавление индикаторов для скальпинга
            df = self._add_scalping_indicators(df)

            # Быстрая проверка EMA кроссовера
            signal_type = self._check_fast_ema_crossover(df)

            if signal_type == "HOLD":
                return self._create_hold_signal(
                    symbol, df["close"].iloc[-1], "Нет кроссовера"
                )

            # Скальпинг фильтры (более мягкие)
            if not self._scalping_filters_passed(df):
                return self._create_hold_signal(
                    symbol, df["close"].iloc[-1], "Скальпинг фильтры не пройдены"
                )

            # Проверка рыночного шума
            if self._is_market_too_noisy(df):
                return self._create_hold_signal(
                    symbol, df["close"].iloc[-1], "Слишком шумный рынок"
                )

            # Расчет уверенности для скальпинга
            confidence = self._calculate_scalping_confidence(df, signal_type)

            # Быстрые TP/SL для скальпинга
            levels = self._calculate_scalping_levels(df, signal_type)

            # Создание сигнала
            signal = TradingSignal(
                symbol=symbol,
                action=signal_type,
                price=df["close"].iloc[-1],
                confidence=confidence,
                indicators=self._extract_scalping_indicators(df),
                timestamp=datetime.now(),
                stop_loss=levels["stop_loss"],
                take_profit=levels["take_profit"],
                risk_reward=levels["risk_reward"],
            )

            # Сохранение сигнала
            await self._save_signal_to_db(signal)

            return signal

        except Exception as e:
            logging.error(f"Ошибка скальпинг анализа {symbol}: {e}")
            return self._create_hold_signal(symbol, 0, f"Ошибка: {e}")

    def _add_scalping_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикаторы для скальпинга"""
        # Быстрые EMA
        df["ema_5"] = self.indicators.calculate_ema(df, self.config.EMA_FAST)
        df["ema_13"] = self.indicators.calculate_ema(df, self.config.EMA_SLOW)

        # Быстрый RSI
        df["rsi_3"] = self.indicators.calculate_rsi(df, self.config.RSI_PERIOD_SHORT)
        df["rsi_7"] = self.indicators.calculate_rsi(df, self.config.RSI_PERIOD_LONG)

        # Быстрый ADX
        df["adx"] = self.indicators.calculate_adx(df, period=7)  # Быстрый ADX

        # Быстрый ATR
        df["atr"] = self.indicators.calculate_atr(df, period=7)  # Быстрый ATR

        # Узкие Bollinger Bands
        bb = self.indicators.calculate_bollinger_bands(
            df, self.config.BB_PERIOD, self.config.BB_STD
        )
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]

        # VWAP для 1-минутного таймфрейма
        df["vwap"] = self.indicators.calculate_vwap(df)

        # Momentum для скальпинга
        df["momentum"] = df["close"].pct_change(periods=3) * 100  # 3-минутный momentum

        return df

    def _check_fast_ema_crossover(self, df: pd.DataFrame) -> str:
        """Быстрая проверка пересечения EMA"""
        if len(df) < 2:
            return "HOLD"

        current_ema5 = df["ema_5"].iloc[-1]
        current_ema13 = df["ema_13"].iloc[-1]
        prev_ema5 = df["ema_5"].iloc[-2]
        prev_ema13 = df["ema_13"].iloc[-2]

        # Проверка на NaN
        if (
            pd.isna(current_ema5)
            or pd.isna(current_ema13)
            or pd.isna(prev_ema5)
            or pd.isna(prev_ema13)
        ):
            return "HOLD"

        # Бычий кроссовер
        if current_ema5 > current_ema13 and prev_ema5 <= prev_ema13:
            return "BUY"

        # Медвежий кроссовер
        if current_ema5 < current_ema13 and prev_ema5 >= prev_ema13:
            return "SELL"

        return "HOLD"

    def _scalping_filters_passed(self, df: pd.DataFrame) -> bool:
        """Мягкие фильтры для скальпинга"""
        current_row = df.iloc[-1]

        # 1. Мягкий ADX фильтр
        if pd.notna(current_row["adx"]) and current_row["adx"] < self.config.MIN_ADX:
            return False

        # 2. Объемный фильтр (более мягкий)
        volume_ratio = current_row["volume"] / df["volume"].rolling(10).mean().iloc[-1]
        if volume_ratio < self.config.MIN_VOLUME_RATIO:
            return False

        # 3. Проверка на слишком узкий спред
        if pd.notna(current_row["bb_upper"]) and pd.notna(current_row["bb_lower"]):
            bb_width = (
                current_row["bb_upper"] - current_row["bb_lower"]
            ) / current_row["bb_middle"]
            if bb_width < self.config.MIN_BB_WIDTH:
                return False

        return True

    def _is_market_too_noisy(self, df: pd.DataFrame) -> bool:
        """Проверка на шумный рынок"""
        # Проверяем последние 10 минут на резкие движения
        recent_data = df.tail(10)
        price_changes = recent_data["close"].pct_change().abs()

        # Если слишком много резких движений - пропускаем
        if (price_changes > 0.005).sum() > 7:  # Более 7 движений > 0.5%
            return True

        return False

    def _calculate_scalping_confidence(
        self, df: pd.DataFrame, signal_type: str
    ) -> float:
        """Расчет уверенности для скальпинга"""
        current_row = df.iloc[-1]
        confidence_factors = []

        # 1. Сила EMA разворота
        ema_spread = (
            abs(current_row["ema_5"] - current_row["ema_13"]) / current_row["close"]
        )
        ema_score = min(ema_spread * 200, 1.0)  # Более чувствительно
        confidence_factors.append(ema_score)

        # 2. Momentum подтверждение
        momentum = current_row.get("momentum", 0)
        if signal_type == "BUY" and momentum > 0:
            momentum_score = min(momentum / 0.5, 1.0)
        elif signal_type == "SELL" and momentum < 0:
            momentum_score = min(abs(momentum) / 0.5, 1.0)
        else:
            momentum_score = 0.3
        confidence_factors.append(momentum_score)

        # 3. Быстрый RSI
        rsi_3 = current_row.get("rsi_3", 50)
        if signal_type == "BUY" and rsi_3 < 40:
            rsi_score = 0.8
        elif signal_type == "SELL" and rsi_3 > 60:
            rsi_score = 0.8
        else:
            rsi_score = 0.4
        confidence_factors.append(rsi_score)

        # 4. Объемное подтверждение
        volume_ratio = current_row["volume"] / df["volume"].rolling(10).mean().iloc[-1]
        volume_score = min(volume_ratio / 1.5, 1.0)
        confidence_factors.append(volume_score)

        return np.mean(confidence_factors)

    def _calculate_scalping_levels(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """Быстрые TP/SL для скальпинга"""
        current_price = df["close"].iloc[-1]
        current_atr = df["atr"].iloc[-1]

        if signal_type == "BUY":
            # Быстрый стоп-лосс
            stop_loss = current_price - (current_atr * self.config.ATR_STOP_MULTIPLIER)
            # Быстрая цель
            take_profit = current_price + (
                current_atr * self.config.ATR_TARGET_MULTIPLIER
            )
        else:  # SELL
            stop_loss = current_price + (current_atr * self.config.ATR_STOP_MULTIPLIER)
            take_profit = current_price - (
                current_atr * self.config.ATR_TARGET_MULTIPLIER
            )

        # Risk/Reward
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": risk_reward,
        }

    def _extract_scalping_indicators(self, df: pd.DataFrame) -> Dict:
        """Извлечение индикаторов для скальпинга"""
        current_row = df.iloc[-1]

        return {
            "ema_5": current_row.get("ema_5"),
            "ema_13": current_row.get("ema_13"),
            "rsi_3": current_row.get("rsi_3"),
            "rsi_7": current_row.get("rsi_7"),
            "adx": current_row.get("adx"),
            "atr": current_row.get("atr"),
            "momentum": current_row.get("momentum"),
            "volume_ratio": current_row["volume"]
            / df["volume"].rolling(10).mean().iloc[-1],
            "bb_position": (
                (
                    (current_row["close"] - current_row["bb_lower"])
                    / (current_row["bb_upper"] - current_row["bb_lower"])
                )
                if pd.notna(current_row.get("bb_upper"))
                else 0.5
            ),
            "close": current_row.get("close"),
        }

    def _create_hold_signal(
        self, symbol: str, price: float, reason: str
    ) -> TradingSignal:
        """Создание HOLD сигнала"""
        return TradingSignal(
            symbol=symbol,
            action="HOLD",
            price=price,
            confidence=0.0,
            indicators={"reason": reason},
            timestamp=datetime.now(),
        )

    async def _save_signal_to_db(self, signal: TradingSignal):
        """Сохранение сигнала в базу данных"""
        signal_data = {
            "timestamp": signal.timestamp,
            "symbol": signal.symbol,
            "action": signal.action,
            "price": signal.price,
            "confidence": signal.confidence,
            "ema_9": signal.indicators.get("ema_5"),  # Маппинг для совместимости с БД
            "ema_21": signal.indicators.get("ema_13"),
            "rsi_14": signal.indicators.get("rsi_7"),
            "adx": signal.indicators.get("adx"),
            "bb_position": signal.indicators.get("bb_position"),
            "vwap_deviation": 0,  # Заглушка для совместимости
            "volume_ratio": signal.indicators.get("volume_ratio"),
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_reward": signal.risk_reward,
            "mode": self.config.TRADING_MODE,
            "indicators": signal.indicators,
        }

        await self.db.save_signal(signal_data)
