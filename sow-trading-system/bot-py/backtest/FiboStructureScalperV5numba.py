import os
import pandas as pd
import talib
import numpy as np
from numba import njit, float64, int64, bool_  # Импорт для Numba
import time
import sqlite3  # Импорт для работы с SQLite
from rich.console import Console  # Импорт для красивого вывода в консоль
from rich.table import Table
from rich.panel import Panel
from rich import box
import matplotlib.pyplot as plt
from math import copysign


# --- КОНСТАНТЫ И ПУТЬ К ФАЙЛУ ---
# ПРИМЕЧАНИЕ: Путь к файлу должен быть корректно установлен в вашей среде
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(
    SCRIPT_DIR,
    "BTC_USDT_1m_2025-07-01_00_00_00_to_2025-09-01_00_00_00.csv",
)

if not os.path.exists(FILE_PATH):
    print(f"Ошибка: Файл {FILE_PATH} не найден!")
    print("Убедитесь, что файл находится в той же папке, что и скрипт.")
    exit()

# Коды для причины закрытия, используемые в Numba
REASON_SL = 1
REASON_TP = 2
REASON_REVERSE_TREND = 3
REASON_HTF_CHANGE = 4
REASON_LIQUIDATION = 5
REASON_END_OF_DATA = 6
REASON_MAP = {
    REASON_SL: "Stop Loss",
    REASON_TP: "Take Profit",
    REASON_REVERSE_TREND: "Reverse Trend",
    REASON_HTF_CHANGE: "HTF Trend Change",
    REASON_LIQUIDATION: "Liquidation",
    REASON_END_OF_DATA: "End of Data",
}

# Определяем структуру данных для NumPy/Pandas
TRADE_DTYPE = np.dtype(
    [
        ("entry_index", np.int32),
        ("exit_index", np.int32),
        ("entry_price", np.float64),
        ("exit_price", np.float64),
        ("pnl", np.float64),
        ("side_code", np.int8),
        ("liquidation", np.bool_),
        ("close_reason_code", np.int8),
        ("equity_after_trade", np.float64),
        ("position_qty", np.float64),
    ]
)

# =========================================================================
# === МОДУЛЬ 1: КОНФИГУРАЦИЯ СТРАТЕГИИ И ПЕРСИСТЕНТНОСТИ ===
# =========================================================================


class StrategyConfig:
    """Хранит все параметры для конкретной стратегии."""

    def __init__(self, **kwargs):
        # 1. Финансовые Параметры и Риск
        self.initial_capital = kwargs.get("initial_capital", 100.0)
        self.leverage = kwargs.get("leverage", 10.0)
        self.target_roi_percent = kwargs.get("target_roi_percent", 20.0)
        self.risk_roi_percent = kwargs.get("risk_roi_percent", 10.0)
        self.maker_fee = kwargs.get("maker_fee", 0.0002)
        self.taker_fee = kwargs.get("taker_fee", 0.0005)

        # 2. Параметры Индикаторов (1m ТФ)
        self.ema_fast_len = kwargs.get("ema_fast_len", 9)
        self.ema_slow_len = kwargs.get("ema_slow_len", 21)
        self.swing_window = kwargs.get("swing_window", 500)
        self.atr_exit_len = kwargs.get("atr_exit_len", 14)
        self.atr_min_multiplier = kwargs.get("atr_min_multiplier", 1.5)
        self.fibo_levels = kwargs.get(
            "fibo_levels", np.array([0.5, 0.618], dtype=np.float64)
        )

        # 3. Параметры Старшего Таймфрейма (HTF Filter)
        self.htf_period = kwargs.get("htf_period", "15min")
        self.htf_ema_fast_len = kwargs.get("htf_ema_fast_len", 9)
        self.htf_ema_slow_len = kwargs.get("htf_ema_slow_len", 21)

        # Расчет производных параметров
        self.required_price_move_for_tp_percent = (
            self.target_roi_percent / self.leverage
        )
        self.required_price_move_for_sl_percent = self.risk_roi_percent / self.leverage
        self.risk_per_trade_percent = self.risk_roi_percent

        print("--- Автоматический Расчет Параметров ---")
        print(
            f"Целевое движение цены для TP: {self.required_price_move_for_tp_percent:.2f}%"
        )
        print(
            f"Макс. допустимое движение цены для SL: {self.required_price_move_for_sl_percent:.2f}%"
        )
        print("-" * 45)


class PersistenceConfig:
    """Хранит настройки сохранения результатов."""

    def __init__(self, **kwargs):
        self.save_to_sqlite = kwargs.get("save_to_sqlite", True)
        self.save_to_csv = kwargs.get("save_to_csv", True)
        self.save_to_txt = kwargs.get("save_to_txt", False)
        self.sqlite_db_name = kwargs.get("sqlite_db_name", "backtest_results.db")
        self.table_name = kwargs.get("table_name", "fibo_trades")
        self.output_file_prefix = kwargs.get("output_file_prefix", "fibo_trades_")


# =========================================================================
# === МОДУЛЬ 2: ПОДГОТОВКА ДАННЫХ И ИНДИКАТОРОВ ===
# =========================================================================


def load_data(file_path):
    """Загрузка, сортировка и базовый анализ данных."""
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
    except Exception as e:
        print(f"Ошибка при чтении файла CSV: {e}")
        exit()

    df = df.sort_values("timestamp").reset_index(drop=True)
    print("--- Загрузка и Анализ Данных ---")
    print(df.info())
    print("\n")
    return df


def calculate_fibo_strategy_indicators(df, config: StrategyConfig):
    """Расчет индикаторов, требуемых для стратегии Fibo Structure Scalper."""

    # 1. EMAs (9 и 21) для определения тренда на 1m
    df["ema_fast"] = talib.EMA(df["close"], timeperiod=config.ema_fast_len)
    df["ema_slow"] = talib.EMA(df["close"], timeperiod=config.ema_slow_len)

    # 2. ATR для фильтрации входов (качество сигнала)
    df["atr_val"] = talib.ATR(
        df["high"], df["low"], df["close"], timeperiod=config.atr_exit_len
    )

    # 3. Расчет структурных точек (Swing High/Low)
    # ПРИМЕЧАНИЕ: Rolling Window Center=True для определения структуры
    df["is_swing_high"] = (
        df["high"] == df["high"].rolling(window=config.swing_window, center=True).max()
    )
    df["is_swing_low"] = (
        df["low"] == df["low"].rolling(window=config.swing_window, center=True).min()
    )

    # 4. Расчет HTF EMAs и их слияние

    # Агрегация данных в HTF (старший таймфрейм)
    df_htf = (
        df.set_index("timestamp")
        .resample(config.htf_period)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    # Расчет HTF EMAs
    df_htf["htf_ema_fast"] = talib.EMA(
        df_htf["close"], timeperiod=config.htf_ema_fast_len
    )
    df_htf["htf_ema_slow"] = talib.EMA(
        df_htf["close"], timeperiod=config.htf_ema_slow_len
    )

    df_htf["htf_is_uptrend"] = df_htf["htf_ema_fast"] > df_htf["htf_ema_slow"]

    df_htf_trend = df_htf[["timestamp", "htf_is_uptrend"]].rename(
        columns={"timestamp": "htf_timestamp"}
    )

    # merge_asof позволяет сопоставить 1m бар с последним завершенным HTF баром
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        df_htf_trend.sort_values("htf_timestamp"),
        left_on="timestamp",
        right_on="htf_timestamp",
        direction="backward",
    ).drop(columns=["htf_timestamp"])

    # Удаляем NaN после всех расчетов
    df = df.dropna().reset_index(drop=True)

    print(f"--- DEBUG: После расчета индикаторов осталось {len(df)} строк.")

    return df


# =========================================================================
# === МОДУЛЬ 3: ЯДРО NUMBA (Остается максимально быстрым) ===
# =========================================================================


# Вспомогательная функция для расчета цены ликвидации (оптимизирована Numba)
@njit(float64(int64, float64, float64, float64, float64))
def _calculate_liquidation_price_numba(
    side_code, entry_price, leverage, maker_fee, taker_fee
):
    """
    Рассчитывает приблизительную цену ликвидации в режиме Numba.

    ВНИМАНИЕ: Формула является приблизительной и учитывает минимальную
    маржу (MMR) и комиссии, которые сдвигают Liq Price ближе к Entry.
    """
    mmr = 0.006  # Минимальная маржа
    # Суммарная комиссия за вход/выход, влияющая на маржу
    total_fee_rate = maker_fee + taker_fee

    if side_code == 1:  # BUY (Long)
        # При покупке цена ликвидации ниже. Комиссии уменьшают знаменатель,
        # что приводит к более агрессивной (ближе к Entry) цене Liq.
        return entry_price * (1.0 - (1.0 / leverage) + mmr) / (1.0 + total_fee_rate)
    else:  # SELL (Short)
        # При продаже цена ликвидации выше. Комиссии уменьшают знаменатель,
        # что приводит к более агрессивной (ближе к Entry) цене Liq.
        # ИСПРАВЛЕНО: Убрана опечатка (двойной taker_fee) и заменена на total_fee_rate.
        return entry_price * (1.0 + (1.0 / leverage) - mmr) / (1.0 - total_fee_rate)


# Основное ядро симуляции, компилируемое Numba
@njit
def simulate_trades_numba_core(
    close_arr,
    high_arr,
    low_arr,
    timestamp_arr,  # Основные данные (timestamp_arr не используется, но сохранен)
    ema_fast_arr,
    ema_slow_arr,
    atr_val_arr,
    htf_is_uptrend_arr,  # Индикаторы
    is_swing_high_arr,
    is_swing_low_arr,  # Структурные точки
    initial_capital,
    leverage,
    maker_fee,
    taker_fee,
    risk_per_trade_percent,
    required_price_move_for_sl_percent,
    required_price_move_for_tp_percent,
    fibo_levels,
    swing_window,
    atr_min_multiplier,
):
    """
    Основной цикл симуляции, полностью оптимизированный Numba.
    """
    current_equity = initial_capital

    # Структура текущей сделки: (Side, EntryPrice, SLPrice, TPPrice, PositionQty, LiquidationPrice, EquityAtEntry, EntryIndex)
    current_trade = None

    # (entry_index, exit_index, entry_price, exit_price, pnl, side_code, is_liquidation, close_reason_code, equity_after, position_qty)
    closed_trades = []

    last_swing_high_price = -1.0
    last_swing_low_price = -1.0
    last_swing_high_index = -1
    last_swing_low_index = -1

    # Трейдинговая комиссия для мгновенного исполнения (SL/TP/Market Exit)
    # Используем TAKER FEE для обоих этапов
    transaction_fee = taker_fee

    data_len = len(close_arr)
    start_index = swing_window  # Начинаем после инициализации swing_window

    # --- ГЛАВНЫЙ ЦИКЛ БЭКТЕСТА ---
    for i in range(max(swing_window, start_index), data_len):
        current_high = high_arr[i]
        current_low = low_arr[i]
        current_close = close_arr[i]

        ema_fast = ema_fast_arr[i]
        ema_slow = ema_slow_arr[i]
        atr_val = atr_val_arr[i]
        htf_is_uptrend = htf_is_uptrend_arr[i]

        # --- 1. Обновление Структурных Точек ---

        if is_swing_high_arr[i] and (
            last_swing_low_index == -1 or i > last_swing_low_index
        ):
            last_swing_high_price = high_arr[i]
            last_swing_high_index = i

        if is_swing_low_arr[i] and (
            last_swing_high_index == -1 or i > last_swing_high_index
        ):
            last_swing_low_price = low_arr[i]
            last_swing_low_index = i

        # --- 2. Расчет Уровней Фибоначчи ---
        fibo_levels_list = []
        is_uptrend = ema_fast > ema_slow
        is_downtrend = ema_fast < ema_slow

        is_long_wave_available = (
            last_swing_high_index > last_swing_low_index and last_swing_low_index != -1
        )
        is_short_wave_available = (
            last_swing_low_index > last_swing_high_index and last_swing_high_index != -1
        )

        if (
            is_long_wave_available
            and last_swing_high_price > 0
            and last_swing_low_price > 0
        ):
            # Волна для LONG (рост от Low до High, ждем коррекции)
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                # Уровень Фибо: High - (Диапазон * Уровень)
                fibo_levels_list.append(last_swing_high_price - (diff * level))

        elif (
            is_short_wave_available
            and last_swing_high_price > 0
            and last_swing_low_price > 0
        ):
            # Волна для SHORT (падение от High до Low, ждем коррекции)
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                # Уровень Фибо: Low + (Диапазон * Уровень)
                fibo_levels_list.append(last_swing_low_price + (diff * level))

        # --- 3. Условия Входа ---
        long_entry_cond = False
        short_entry_cond = False

        if current_trade is None:

            # Вход LONG: 1m Up + HTF Up + Откат к Фибо (закрываемся выше)
            if is_uptrend and is_long_wave_available and htf_is_uptrend:
                for fibo_level in fibo_levels_list:
                    if (
                        fibo_level > current_low
                        and fibo_level < current_high
                        and current_close > fibo_level  # Закрытие выше уровня Фибо
                    ):
                        long_entry_cond = True
                        break

            # Вход SHORT: 1m Down + HTF Down + Откат к Фибо (закрываемся ниже)
            if is_downtrend and is_short_wave_available and not htf_is_uptrend:
                for fibo_level in fibo_levels_list:
                    if (
                        fibo_level > current_low
                        and fibo_level < current_high
                        and current_close < fibo_level  # Закрытие ниже уровня Фибо
                    ):
                        short_entry_cond = True
                        break

        # === Логика Закрытия Сделки ===
        if current_trade is not None:
            # Распаковка кортежа
            (
                side_code,
                entry_price,
                stop_loss_price,
                take_profit_price,
                position_qty,
                liquidation_price,
                equity_at_entry,
                entry_index,
            ) = current_trade

            exit_price = 0.0
            close_reason_code = 0
            is_liquidation = False

            # Проверки срабатывания

            # 1. Stop Loss
            if (side_code == 1 and current_low <= stop_loss_price) or (
                side_code == -1 and current_high >= stop_loss_price
            ):
                exit_price = stop_loss_price
                close_reason_code = REASON_SL
            # 2. Take Profit
            elif (side_code == 1 and current_high >= take_profit_price) or (
                side_code == -1 and current_low <= take_profit_price
            ):
                exit_price = take_profit_price
                close_reason_code = REASON_TP
            # 3. Ликвидация (Должна быть проверена до других выходов по рынку)
            elif (side_code == 1 and current_low <= liquidation_price) or (
                side_code == -1 and current_high >= liquidation_price
            ):
                exit_price = liquidation_price
                close_reason_code = REASON_LIQUIDATION
                is_liquidation = True
            # 4. Обратный тренд (1m EMA crossover)
            elif (side_code == 1 and is_downtrend) or (side_code == -1 and is_uptrend):
                exit_price = current_close
                close_reason_code = REASON_REVERSE_TREND
            # 5. Смена HTF тренда (Фильтр)
            elif (side_code == 1 and not htf_is_uptrend) or (
                side_code == -1 and htf_is_uptrend
            ):
                exit_price = current_close
                close_reason_code = REASON_HTF_CHANGE

            if close_reason_code != 0:
                # --- Расчет PNL ---
                qty = position_qty

                # Расчет общей комиссии (USD)
                # Используем Taker Fee для входа и выхода.
                fee_entry_usd = qty * entry_price * transaction_fee
                fee_exit_usd = qty * exit_price * transaction_fee
                total_fees = fee_entry_usd + fee_exit_usd

                if side_code == 1:  # BUY
                    pnl = (exit_price - entry_price) * qty * leverage - total_fees
                else:  # SELL
                    pnl = (entry_price - exit_price) * qty * leverage - total_fees

                current_equity += pnl

                # Сохраняем сделку
                closed_trades.append(
                    (
                        entry_index,
                        i,
                        entry_price,
                        exit_price,
                        pnl,
                        side_code,
                        is_liquidation,
                        close_reason_code,
                        current_equity,
                        position_qty,
                    )
                )
                current_trade = None

        # === Логика Открытия Сделки ===
        if current_trade is None and (long_entry_cond or short_entry_cond):

            # Расчет риска и размера позиции
            sl_price_dist_usd = current_close * (
                required_price_move_for_sl_percent / 100.0
            )
            min_sl_dist_by_atr = atr_val * atr_min_multiplier
            max_loss_usd = current_equity * (risk_per_trade_percent / 100.0)

            if sl_price_dist_usd > 0.0:
                # Размер позиции на основе риска: (Max Loss) / (Риск в $ на 1 Qty * Leverage)
                risk_based_qty = max_loss_usd / (sl_price_dist_usd * leverage)
                # Максимальный размер позиции по марже: (Equity * Leverage) / Current Price
                margin_based_qty = (current_equity * leverage) / current_close
                position_qty = min(risk_based_qty, margin_based_qty)
            else:
                position_qty = 0.0

            # Проверка качества: SL должен быть больше минимального ATR
            quality_check_ok = sl_price_dist_usd >= min_sl_dist_by_atr

            # Проверка комиссий: общая комиссия не должна превышать 50% от максимального риска
            # Total Fees = 2 * (Qty * Price * TakerFee) - приблизительно
            estimated_total_commission = (
                position_qty * current_close * transaction_fee * 2.0
            )
            commission_check_ok = estimated_total_commission < (max_loss_usd * 0.5)

            if position_qty > 0.0 and quality_check_ok and commission_check_ok:

                if long_entry_cond:
                    side_code = 1  # BUY
                    entry_price = current_close
                    lq_price = _calculate_liquidation_price_numba(
                        side_code, entry_price, leverage, maker_fee, taker_fee
                    )

                    stop_loss_price = entry_price * (
                        1.0 - (required_price_move_for_sl_percent / 100.0)
                    )
                    take_profit_price = entry_price * (
                        1.0 + (required_price_move_for_tp_percent / 100.0)
                    )

                    current_trade = (
                        side_code,
                        entry_price,
                        stop_loss_price,
                        take_profit_price,
                        position_qty,
                        lq_price,
                        current_equity,
                        i,
                    )

                elif short_entry_cond:
                    side_code = -1  # SELL
                    entry_price = current_close
                    lq_price = _calculate_liquidation_price_numba(
                        side_code, entry_price, leverage, maker_fee, taker_fee
                    )

                    stop_loss_price = entry_price * (
                        1.0 + (required_price_move_for_sl_percent / 100.0)
                    )
                    take_profit_price = entry_price * (
                        1.0 - (required_price_move_for_tp_percent / 100.0)
                    )

                    current_trade = (
                        side_code,
                        entry_price,
                        stop_loss_price,
                        take_profit_price,
                        position_qty,
                        lq_price,
                        current_equity,
                        i,
                    )

    # Закрытие последней сделки, если она осталась открытой
    if current_trade is not None:
        side_code, entry_price, _, _, position_qty, _, _, entry_index = current_trade

        exit_price = close_arr[-1]
        exit_index = data_len - 1

        # Расчет комиссии (USD)
        fee_entry_usd = position_qty * entry_price * transaction_fee
        fee_exit_usd = position_qty * exit_price * transaction_fee
        total_fees = fee_entry_usd + fee_exit_usd

        if side_code == 1:  # BUY
            pnl = (exit_price - entry_price) * position_qty * leverage - total_fees
        else:  # SELL
            pnl = (entry_price - exit_price) * position_qty * leverage - total_fees

        current_equity += pnl

        closed_trades.append(
            (
                entry_index,
                exit_index,
                entry_price,
                exit_price,
                pnl,
                side_code,
                False,  # Не ликвидация
                REASON_END_OF_DATA,
                current_equity,
                position_qty,
            )
        )
        current_trade = None

    return closed_trades, current_equity


# =========================================================================
# === МОДУЛЬ 4: ДРАЙВЕР И ПРЕОБРАЗОВАНИЕ РЕЗУЛЬТАТОВ ===
# =========================================================================


def run_backtest(df, config: StrategyConfig):
    """
    Драйвер для Numba-симуляции: Подготавливает данные и запускает ядро.
    """

    # Подготовка данных (преобразование в массивы NumPy)
    close_arr = df["close"].to_numpy(dtype=np.float64)
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr = df["low"].to_numpy(dtype=np.float64)
    timestamp_arr = df["timestamp"].to_numpy()
    ema_fast_arr = df["ema_fast"].to_numpy(dtype=np.float64)
    ema_slow_arr = df["ema_slow"].to_numpy(dtype=np.float64)
    atr_val_arr = df["atr_val"].to_numpy(dtype=np.float64)
    htf_is_uptrend_arr = df["htf_is_uptrend"].to_numpy(dtype=np.bool_)
    is_swing_high_arr = df["is_swing_high"].to_numpy(dtype=np.bool_)
    is_swing_low_arr = df["is_swing_low"].to_numpy(dtype=np.bool_)

    # Запуск Numba-ядра
    closed_trades_tuples, final_equity = simulate_trades_numba_core(
        close_arr,
        high_arr,
        low_arr,
        timestamp_arr,
        ema_fast_arr,
        ema_slow_arr,
        atr_val_arr,
        htf_is_uptrend_arr,
        is_swing_high_arr,
        is_swing_low_arr,
        config.initial_capital,
        config.leverage,
        config.maker_fee,
        config.taker_fee,
        config.risk_per_trade_percent,
        config.required_price_move_for_sl_percent,
        config.required_price_move_for_tp_percent,
        config.fibo_levels,
        config.swing_window,
        config.atr_min_multiplier,
    )

    # Преобразование Numba-результатов в Pandas DataFrame
    if not closed_trades_tuples:
        return pd.DataFrame(), final_equity

    trades_array = np.array(closed_trades_tuples, dtype=TRADE_DTYPE)
    trades_df = pd.DataFrame(trades_array)

    # Обратное преобразование кодов в понятные значения
    trades_df["side"] = trades_df["side_code"].apply(
        lambda x: "BUY" if x == 1 else "SELL"
    )
    trades_df["close_reason"] = trades_df["close_reason_code"].map(REASON_MAP)

    # Привязка индексов обратно к временным меткам
    trades_df["entry_time"] = df.loc[trades_df["entry_index"], "timestamp"].values
    trades_df["exit_time"] = df.loc[trades_df["exit_index"], "timestamp"].values

    # Удаление служебных колонок
    trades_df = trades_df.drop(
        columns=["side_code", "close_reason_code", "entry_index", "exit_index"]
    )

    return trades_df, final_equity


# =========================================================================
# === МОДУЛЬ 5: ПЕРСИСТЕНТНОСТЬ (SQLite, CSV, TXT) ===
# =========================================================================


def persist_results(trades_df, config: PersistenceConfig):
    """Сохраняет результаты сделок в указанные форматы."""
    if trades_df.empty:
        print("Нет сделок для сохранения.")
        return

    file_prefix = config.output_file_prefix

    # --- 1. Сохранение в SQLite3 ---
    if config.save_to_sqlite:
        db_path = os.path.join(SCRIPT_DIR, config.sqlite_db_name)
        try:
            conn = sqlite3.connect(db_path)
            # Сохраняем сделку вместе с метаданными (имя стратегии, дата и т.д.)
            trades_df.to_sql(config.table_name, conn, if_exists="append", index=False)
            conn.close()
            print(
                f"[PERSIST] Успешно сохранено в SQLite: {config.sqlite_db_name} (Таблица: {config.table_name})"
            )
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в SQLite: {e}")

    # --- 2. Сохранение в CSV ---
    if config.save_to_csv:
        csv_path = os.path.join(
            SCRIPT_DIR, f"{file_prefix}{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        try:
            trades_df.to_csv(csv_path, index=False)
            print(f"[PERSIST] Успешно сохранено в CSV: {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в CSV: {e}")

    # --- 3. Сохранение в TXT (Простой лог) ---
    if config.save_to_txt:
        txt_path = os.path.join(
            SCRIPT_DIR, f"{file_prefix}{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        try:
            log_cols = [
                "entry_time",
                "side",
                "entry_price",
                "exit_price",
                "pnl",
                "close_reason",
                "equity_after_trade",
            ]
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("--- ЛОГ СДЕЛОК БЭКТЕСТА ---\n")
                f.write(trades_df[log_cols].to_string())
            print(f"[PERSIST] Успешно сохранено в TXT: {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить в TXT: {e}")


# =========================================================================
# === МОДУЛЬ 6: АНАЛИЗ И ОТЧЕТНОСТЬ (RICH DASHBOARD) ===
# =========================================================================


def calculate_metrics(trades_df, initial_capital, final_equity):
    """Рассчитывает ключевые метрики бэктеста."""
    if trades_df.empty:
        # Если сделок нет, возвращаем пустые метрики
        return {}, pd.Series([0.0]), pd.Series([initial_capital])

    total_pnl = trades_df["pnl"].sum()
    success_rate = (trades_df["pnl"] > 0).mean() * 100
    liquidation_rate = trades_df["liquidation"].mean() * 100
    num_trades = len(trades_df)

    return_on_capital = (final_equity - initial_capital) / initial_capital * 100

    # Расчет Максимальной Просадки (Max Drawdown - MDD)
    equity_curve = trades_df["equity_after_trade"]
    # Для расчета MDD нужно включить начальный капитал
    full_equity_curve = pd.concat([pd.Series([initial_capital]), equity_curve])
    peak = full_equity_curve.expanding().max()
    drawdown = (peak - full_equity_curve) / peak
    max_drawdown = drawdown.max() * 100 if not drawdown.empty else 0.0

    # MDD без включения стартового капитала (для графика)
    drawdown_for_plot = (
        equity_curve.expanding().max() - equity_curve
    ) / equity_curve.expanding().max()

    metrics = {
        "Начальный капитал": f"{initial_capital:,.2f}",
        "Финальный капитал": f"{final_equity:,.2f}",
        "Общий PNL": f"{total_pnl:,.2f}",
        "ROI (Прибыльность)": f"{return_on_capital:.2f}%",
        "Максимальная просадка (MDD)": f"{max_drawdown:.2f}%",
        "Общее количество сделок": f"{num_trades}",
        "Процент прибыльных сделок": f"{success_rate:.1f}%",
        "Процент ликвидаций": f"{liquidation_rate:.1f}%",
    }

    return metrics, drawdown_for_plot, equity_curve


def display_results_rich(metrics, trades_df, execution_time):
    """Отображает результаты бэктеста в виде консольного дашборда (Rich)."""
    console = Console()

    # 1. Заголовок
    console.rule("[bold cyan]РЕЗУЛЬТАТЫ БЭКТЕСТА (Fibo Structure Scalper)[/bold cyan]")

    # 2. Таблица Основных Метрик
    table = Table(
        title="[bold white]СВОДКА МЕТРИК[/bold white]",
        show_lines=True,
        box=box.ROUNDED,
        style="dim",
    )
    table.add_column("Метрика", style="cyan", justify="left")
    table.add_column("Значение", style="yellow", justify="right")

    # Перевод ROI и PNL в float для проверки знака
    total_pnl = float(metrics.get("Общий PNL", "0").replace("$", "").replace(",", ""))

    for key, value in metrics.items():
        style = (
            "green"
            if ("ROI" in key or "PNL" in key) and total_pnl >= 0
            else (
                "red"
                if ("Просадка" in key or "Ликвидаций" in key or total_pnl < 0)
                else "white"
            )
        )
        table.add_row(key, f"[{style}]{value}[/{style}]")

    table.add_row("---", "---")
    table.add_row(
        "[bold]Время выполнения (Numba)[/bold]",
        f"[magenta]{execution_time:.4f} сек.[/magenta]",
    )

    console.print(
        Panel(table, title="[bold green]ДАШБОРД[/bold green]", border_style="green")
    )

    # 3. Лог Сделок (Первые 5 / Последние 5)
    if not trades_df.empty:
        log_cols = [
            "entry_time",
            "side",
            "entry_price",
            "exit_price",
            "pnl",
            "close_reason",
            "equity_after_trade",
        ]

        log_table = Table(
            title=f"[bold white]ЛОГ ЗАКРЫТЫХ СДЕЛОК (Показано 10 из {len(trades_df)})[/bold white]",
            box=box.MINIMAL,
        )
        for col in log_cols:
            log_table.add_column(
                col.replace("_", " ").capitalize(),
                style="dim",
                justify=(
                    "right"
                    if col not in ["close_reason", "side", "entry_time"]
                    else "left"
                ),
            )

        log_df = pd.concat([trades_df[log_cols].head(5), trades_df[log_cols].tail(5)])

        for _, row in log_df.iterrows():
            pnl_style = "green" if row["pnl"] >= 0 else "red"
            log_table.add_row(
                str(row["entry_time"])[:19],
                f"[bold blue]{row['side']}[/bold blue]",
                f"{row['entry_price']:,.2f}",
                f"{row['exit_price']:,.2f}",
                f"[{pnl_style}]{row['pnl']:,.2f}[/{pnl_style}]",
                f"[dim white]{row['close_reason']}[/dim white]",
                f"[bold]{row['equity_after_trade']:,.2f}[/bold]",
            )

        console.print(log_table)

    # 4. Графики (Примечание для пользователя)
    console.print(
        Panel(
            "[bold yellow]Графики (Equity Curve, Drawdown) будут отображены в отдельном окне Matplotlib.[/bold yellow]",
            border_style="yellow",
        )
    )


# =========================================================================
# === ОСНОВНОЙ СКРИПТ: ИНТЕГРАЦИЯ ===
# =========================================================================

if __name__ == "__main__":

    # 1. ИНИЦИАЛИЗАЦИЯ КОНФИГУРАЦИЙ
    strategy_config = StrategyConfig(
        initial_capital=10.0,
        leverage=4.0,
        target_roi_percent=30.0,
        risk_roi_percent=30.0,
        # ema_fast_len=9, # Можно переопределять любые параметры здесь
    )

    # --- ПРИМЕРЫ КОНФИГУРАЦИЙ ---

    # 1. SCALP_DEFAULT: Агрессивный скальпинг
    # Низкий процент прибыли/риска, высокая частота сделок, умеренное плечо.
    SCALP_DEFAULT = StrategyConfig(
        initial_capital=10.0,
        leverage=5.0,
        target_roi_percent=20.0,  # Цель: +20% от позиции (быстрое закрытие)
        risk_roi_percent=10.0,  # Риск: -10% от позиции (соотношение R:R 2:1)
        ema_fast_len=7,  # Очень быстрая EMA
        ema_slow_len=25,  # Более быстрая медленная EMA
        rsi_period=14,
        fibo_depth=100,
    )

    # 2. BALANCED_DAYTRADE: Сбалансированная внутридневная торговля
    # Умеренное соотношение риска к прибыли (1:1), среднее плечо, стандартные индикаторы.
    BALANCED_DAYTRADE = StrategyConfig(
        initial_capital=100.0,
        leverage=3.0,
        target_roi_percent=30.0,  # Цель: +30%
        risk_roi_percent=30.0,  # Риск: -30% (соотношение R:R 1:1)
        ema_fast_len=12,  # Стандартная EMA
        ema_slow_len=50,  # Стандартная медленная EMA
        rsi_period=14,
        fibo_depth=150,  # Средняя глубина поиска структуры
    )

    # 3. SWING_LOW_LEVERAGE: Свинг-трейдинг с низким риском
    # Низкое плечо, широкие цели и стопы, медленные индикаторы для фильтрации шума.
    SWING_LOW_LEVERAGE = StrategyConfig(
        initial_capital=500.0,
        leverage=1.5,
        target_roi_percent=50.0,  # Цель: +50% (долгий ход)
        risk_roi_percent=25.0,  # Риск: -25% (соотношение R:R 2:1)
        ema_fast_len=20,  # Медленная EMA
        ema_slow_len=100,  # Очень медленная EMA (для сильного тренда)
        rsi_period=21,  # Более длинный RSI
        fibo_depth=300,  # Глубокий поиск структуры
    )

    # 4. HIGH_RISK_HIGH_REWARD: Высокий риск / Высокая награда
    # Агрессивное плечо и очень выгодное соотношение R:R (3:1) в расчете на сильное движение.
    HIGH_RISK_HIGH_REWARD = StrategyConfig(
        initial_capital=50.0,
        leverage=8.0,
        target_roi_percent=45.0,  # Цель: +45%
        risk_roi_percent=15.0,  # Риск: -15% (соотношение R:R 3:1)
        ema_fast_len=9,
        ema_slow_len=30,
        rsi_period=10,  # Более чувствительный RSI
        fibo_depth=120,
    )

    # 5. CONSERVATIVE_CONFIRMATION: Консервативный с фильтром
    # Упор на подтверждение долгосрочного тренда (высокое значение ema_slow_len)
    CONSERVATIVE_CONFIRMATION = StrategyConfig(
        initial_capital=200.0,
        leverage=2.0,
        target_roi_percent=25.0,
        risk_roi_percent=20.0,
        ema_fast_len=15,
        ema_slow_len=200,  # Очень длинный период для фильтрации
        rsi_period=14,
        fibo_depth=200,
    )

    # --- ПРИМЕР ИСПОЛЬЗОВАНИЯ В ВАШЕМ СКРИПТЕ ---

    # Выберите одну из конфигураций для запуска бэктеста:
    strategy_config = HIGH_RISK_HIGH_REWARD

    persistence_config = PersistenceConfig(
        save_to_sqlite=True,
        save_to_csv=False,
        save_to_txt=False,  # Отключено по умолчанию
        table_name="fibo_scalper_run_1",  # Указываем конкретную таблицу для этого прогона
    )

    # 2. ПОДГОТОВКА ДАННЫХ
    df_raw = load_data(FILE_PATH)
    df = calculate_fibo_strategy_indicators(df_raw.copy(), strategy_config)

    # 3. РАЗОГРЕВ NUMBA (Компиляция)
    print("--- DEBUG: Запуск 'разогрева' Numba (компиляция)...")
    try:
        # Для разогрева достаточно передать небольшой кусок данных
        _, _ = run_backtest(
            df.head(strategy_config.swing_window + 100).copy(), strategy_config
        )
        print("--- DEBUG: Компиляция завершена. Измерение скорости...")
    except Exception as e:
        print(f"[ERROR] Ошибка при разогреве Numba: {e}")
        # Продолжаем, но с ошибкой разогрева
        pass

    # 4. ЗАПУСК БЭКТЕСТА И ИЗМЕРЕНИЕ ВРЕМЕНИ
    start_time = time.perf_counter()
    trades_df, final_equity = run_backtest(df, strategy_config)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # 5. АНАЛИЗ И ОТЧЕТНОСТЬ
    metrics, drawdown_data, equity_curve_data = calculate_metrics(
        trades_df, strategy_config.initial_capital, final_equity
    )

    # Вывод Rich Dashboard
    display_results_rich(metrics, trades_df, execution_time)

    # Построение графиков
    if not trades_df.empty:
        # Для построения графиков используем исходную функцию
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # --- График 1: Кривая Эквити ---
        axes[0].plot(
            equity_curve_data.index,
            equity_curve_data,
            label="Кривая Эквити",
            color="green",
        )
        axes[0].axhline(
            strategy_config.initial_capital,
            color="grey",
            linestyle="--",
            label="Начальный Капитал",
        )
        axes[0].set_title("1. Кривая Эквити (Equity Curve)", fontsize=14)
        axes[0].set_xlabel("Номер Сделки")
        axes[0].set_ylabel("Капитал ($)")
        axes[0].grid(True, alpha=0.5)
        axes[0].legend()

        # --- График 2: Максимальная Просадка ---
        # Drawdown Data имеет на 1 элемент больше, так как включает стартовый капитал
        axes[1].fill_between(
            drawdown_data.index,
            drawdown_data * 100,
            color="red",
            alpha=0.3,
            label="Просадка (%)",
        )
        axes[1].set_title("2. Максимальная Просадка (Max Drawdown)", fontsize=14)
        axes[1].set_xlabel("Номер Сделки")
        axes[1].set_ylabel("Просадка (%)")
        axes[1].grid(True, alpha=0.5)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    # 6. ПЕРСИСТЕНТНОСТЬ (Сохранение)
    persist_results(trades_df, persistence_config)
