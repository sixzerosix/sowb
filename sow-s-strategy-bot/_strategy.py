import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from indicators import TechnicalIndicators
import logging


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


class EnhancedStrategyS:
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
        self.indicators = TechnicalIndicators()

    async def analyze_symbol(self, symbol: str) -> TradingSignal:
        """Главный метод анализа символа"""
        try:
            # Получение данных
            df = await self.db.get_ohlcv(symbol, self.config.TIMEFRAME, 500)

            if df.empty or len(df) < 100:
                return self._create_hold_signal(symbol, 0, "Недостаточно данных")

            # Добавление индикаторов
            df = self.indicators.add_all_indicators(df, self.config)

            # Проверка базового кроссовера EMA
            signal_type = self._check_ema_crossover(df)

            if signal_type == "HOLD":
                return self._create_hold_signal(
                    symbol, df["close"].iloc[-1], "Нет кроссовера EMA"
                )

            # Применение фильтров качества
            if not self._quality_filters_passed(df):
                return self._create_hold_signal(
                    symbol, df["close"].iloc[-1], "Фильтры качества не пройдены"
                )

            # Расчет уверенности в сигнале
            confidence = self._calculate_confidence(df, signal_type)

            # Расчет динамических TP/SL
            levels = self._calculate_dynamic_levels(df, signal_type)

            # Создание сигнала
            signal = TradingSignal(
                symbol=symbol,
                action=signal_type,
                price=df["close"].iloc[-1],
                confidence=confidence,
                indicators=self._extract_indicators(df),
                timestamp=datetime.now(),
                stop_loss=levels["stop_loss"],
                take_profit=levels["take_profit"],
                risk_reward=levels["risk_reward"],
            )

            # Сохранение сигнала в БД
            await self._save_signal_to_db(signal)

            return signal

        except Exception as e:
            logging.error(f"Ошибка анализа {symbol}: {e}")
            return self._create_hold_signal(symbol, 0, f"Ошибка: {e}")

    def _check_ema_crossover(self, df: pd.DataFrame) -> str:
        """Проверка пересечения EMA"""
        if len(df) < 2:
            return "HOLD"

        current_ema9 = df["ema_9"].iloc[-1]
        current_ema21 = df["ema_21"].iloc[-1]
        prev_ema9 = df["ema_9"].iloc[-2]
        prev_ema21 = df["ema_21"].iloc[-2]

        # Проверка на NaN
        if (
            pd.isna(current_ema9)
            or pd.isna(current_ema21)
            or pd.isna(prev_ema9)
            or pd.isna(prev_ema21)
        ):
            return "HOLD"

        # Бычий кроссовер
        if current_ema9 > current_ema21 and prev_ema9 <= prev_ema21:
            return "BUY"

        # Медвежий кроссовер
        if current_ema9 < current_ema21 and prev_ema9 >= prev_ema21:
            return "SELL"

        return "HOLD"

    def _quality_filters_passed(self, df: pd.DataFrame) -> bool:
        """Проверка фильтров качества сигнала"""
        current_row = df.iloc[-1]

        # 1. Фильтр силы тренда (ADX)
        if pd.notna(current_row["adx"]) and current_row["adx"] < self.config.MIN_ADX:
            logging.debug(f"ADX фильтр не пройден: {current_row['adx']}")
            return False

        # 2. Фильтр объема
        volume_ratio = current_row["volume"] / df["volume"].rolling(20).mean().iloc[-1]
        if volume_ratio < self.config.MIN_VOLUME_RATIO:
            logging.debug(f"Объемный фильтр не пройден: {volume_ratio}")
            return False

        # 3. Фильтр волатильности (Bollinger Bands)
        if pd.notna(current_row["bb_upper"]) and pd.notna(current_row["bb_lower"]):
            bb_width = (
                current_row["bb_upper"] - current_row["bb_lower"]
            ) / current_row["bb_middle"]
            if bb_width < self.config.MIN_BB_WIDTH:
                logging.debug(f"BB ширина фильтр не пройден: {bb_width}")
                return False

        # 4. RSI фильтр (избегаем экстремальных зон)
        if pd.notna(current_row["rsi_14"]):
            if (
                current_row["rsi_14"] > self.config.RSI_OVERBOUGHT
                or current_row["rsi_14"] < self.config.RSI_OVERSOLD
            ):
                logging.debug(f"RSI фильтр не пройден: {current_row['rsi_14']}")
                return False

        return True

    def _calculate_confidence(self, df: pd.DataFrame, signal_type: str) -> float:
        """Расчет уверенности в сигнале"""
        current_row = df.iloc[-1]
        confidence_factors = []

        # 1. Сила тренда (ADX)
        if pd.notna(current_row["adx"]):
            adx_score = min(current_row["adx"] / 50, 1.0)
            confidence_factors.append(adx_score)

        # 2. Объемное подтверждение
        volume_ratio = current_row["volume"] / df["volume"].rolling(20).mean().iloc[-1]
        volume_score = min(volume_ratio / 2, 1.0)
        confidence_factors.append(volume_score)

        # 3. Дивергенция EMA (чем больше расстояние, тем сильнее сигнал)
        if pd.notna(current_row["ema_9"]) and pd.notna(current_row["ema_21"]):
            ema_spread = (
                abs(current_row["ema_9"] - current_row["ema_21"]) / current_row["close"]
            )
            ema_score = min(ema_spread * 100, 1.0)
            confidence_factors.append(ema_score)

        # 4. VWAP подтверждение
        if pd.notna(current_row["vwap"]):
            vwap_alignment = (
                1.0
                if (
                    (
                        signal_type == "BUY"
                        and current_row["close"] > current_row["vwap"]
                    )
                    or (
                        signal_type == "SELL"
                        and current_row["close"] < current_row["vwap"]
                    )
                )
                else 0.5
            )
            confidence_factors.append(vwap_alignment)

        # 5. RSI momentum
        if pd.notna(current_row["rsi_14"]):
            if signal_type == "BUY":
                rsi_score = max(0, (current_row["rsi_14"] - 50) / 20)
            else:  # SELL
                rsi_score = max(0, (50 - current_row["rsi_14"]) / 20)
            confidence_factors.append(min(rsi_score, 1.0))

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _calculate_dynamic_levels(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """Расчет динамических TP/SL уровней"""
        current_price = df["close"].iloc[-1]
        current_atr = df["atr"].iloc[-1]

        # Fibonacci уровни
        fib_levels = self.indicators.calculate_fibonacci_levels(
            df, self.config.FIBONACCI_LOOKBACK
        )

        if signal_type == "BUY":
            # Stop Loss: ниже текущей цены
            atr_stop = current_price - (current_atr * self.config.ATR_STOP_MULTIPLIER)

            # Найти ближайший уровень поддержки Fibonacci
            support_levels = [
                level for level in fib_levels.values() if level < current_price
            ]
            nearest_support = (
                max(support_levels) if support_levels else current_price * 0.98
            )

            stop_loss = max(atr_stop, nearest_support)

            # Take Profit: выше текущей цены
            atr_target = current_price + (
                current_atr * self.config.ATR_TARGET_MULTIPLIER
            )

            # Найти ближайший уровень сопротивления Fibonacci
            resistance_levels = [
                level for level in fib_levels.values() if level > current_price
            ]
            nearest_resistance = (
                min(resistance_levels) if resistance_levels else current_price * 1.02
            )

            take_profit = min(atr_target, nearest_resistance)

        else:  # SELL
            # Stop Loss: выше текущей цены
            atr_stop = current_price + (current_atr * self.config.ATR_STOP_MULTIPLIER)

            # Найти ближайший уровень сопротивления Fibonacci
            resistance_levels = [
                level for level in fib_levels.values() if level > current_price
            ]
            nearest_resistance = (
                min(resistance_levels) if resistance_levels else current_price * 1.02
            )

            stop_loss = min(atr_stop, nearest_resistance)

            # Take Profit: ниже текущей цены
            atr_target = current_price - (
                current_atr * self.config.ATR_TARGET_MULTIPLIER
            )

            # Найти ближайший уровень поддержки Fibonacci
            support_levels = [
                level for level in fib_levels.values() if level < current_price
            ]
            nearest_support = (
                max(support_levels) if support_levels else current_price * 0.98
            )

            take_profit = max(atr_target, nearest_support)

        # Расчет Risk/Reward соотношения
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": risk_reward,
        }

    def _extract_indicators(self, df: pd.DataFrame) -> Dict:
        """Извлечение значений индикаторов"""
        current_row = df.iloc[-1]

        indicators = {
            "ema_9": current_row.get("ema_9"),
            "ema_21": current_row.get("ema_21"),
            "rsi_5": current_row.get("rsi_5"),
            "rsi_14": current_row.get("rsi_14"),
            "adx": current_row.get("adx"),
            "atr": current_row.get("atr"),
            "bb_upper": current_row.get("bb_upper"),
            "bb_middle": current_row.get("bb_middle"),
            "bb_lower": current_row.get("bb_lower"),
            "vwap": current_row.get("vwap"),
            "obv": current_row.get("obv"),
            "volume": current_row.get("volume"),
            "close": current_row.get("close"),
        }

        # Дополнительные расчетные метрики
        if pd.notna(current_row.get("bb_upper")) and pd.notna(
            current_row.get("bb_lower")
        ):
            bb_position = (current_row["close"] - current_row["bb_lower"]) / (
                current_row["bb_upper"] - current_row["bb_lower"]
            )
            indicators["bb_position"] = bb_position

        if pd.notna(current_row.get("vwap")):
            vwap_deviation = (current_row["close"] - current_row["vwap"]) / current_row[
                "vwap"
            ]
            indicators["vwap_deviation"] = vwap_deviation

        volume_ratio = current_row["volume"] / df["volume"].rolling(20).mean().iloc[-1]
        indicators["volume_ratio"] = volume_ratio

        return indicators

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
            "ema_9": signal.indicators.get("ema_9"),
            "ema_21": signal.indicators.get("ema_21"),
            "rsi_14": signal.indicators.get("rsi_14"),
            "adx": signal.indicators.get("adx"),
            "bb_position": signal.indicators.get("bb_position"),
            "vwap_deviation": signal.indicators.get("vwap_deviation"),
            "volume_ratio": signal.indicators.get("volume_ratio"),
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_reward": signal.risk_reward,
            "mode": self.config.TRADING_MODE,
            "indicators": signal.indicators,
        }

        await self.db.save_signal(signal_data)
