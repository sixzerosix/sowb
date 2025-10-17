import os
import pandas as pd
import talib
import numpy as np
from numba import njit, float64, int64  # Импорт для Numba

# Импортируем библиотеку для графиков
import matplotlib.pyplot as plt
from math import copysign

# Определяем путь к файлу
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    script_dir,
    "BTC_USDT_1m_2025-07-01_00_00_00_to_2025-09-01_00_00_00.csv",
)

if not os.path.exists(file_path):
    print(f"Ошибка: Файл {file_path} не найден!")
    print("Убедитесь, что файл находится в той же папке, что и скрипт.")
    exit()


# === ЭТАП 1: Загрузка и анализ данных (Pandas - без изменений) ===
def load_and_analyze_data(file_path):
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


# === ЭТАП 2: Расчет Индикаторов (Pandas/TA-Lib - без изменений) ===
def calculate_fibo_structure_indicators(
    df, ema_fast_len, ema_slow_len, atr_len, swing_window
):
    """Расчет индикаторов и ключевых структурных точек для стратегии FSS (на 1m TF)."""

    # 1. EMAs (9 и 21) для определения тренда на 1m
    df["ema_fast"] = talib.EMA(df["close"], timeperiod=ema_fast_len)
    df["ema_slow"] = talib.EMA(df["close"], timeperiod=ema_slow_len)

    # 2. ATR для фильтрации входов (качество сигнала)
    df["atr_val"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=atr_len)

    # 3. Расчет структурных точек (Swing High/Low)
    df["is_swing_high"] = (
        df["high"] == df["high"].rolling(window=swing_window, center=True).max()
    )
    df["is_swing_low"] = (
        df["low"] == df["low"].rolling(window=swing_window, center=True).min()
    )

    # Удаляем NaN
    df = df.dropna().reset_index(drop=True)

    print(f"--- DEBUG: После расчета индикаторов осталось {len(df)} строк.")

    return df


# =========================================================================
# === Numba-оптимизированные Функции ===
# =========================================================================


# Вспомогательная функция для расчета цены ликвидации (оптимизирована Numba)
@njit(float64(int64, float64, float64, float64, float64))
def _calculate_liquidation_price_numba(
    side_code, entry_price, leverage, maker_fee, taker_fee
):
    """Рассчитывает приблизительную цену ликвидации в режиме Numba.
    side_code: 1 (BUY), -1 (SELL)
    """
    mmr = 0.006
    if side_code == 1:  # BUY
        # При покупке цена ликвидации ниже
        return (
            entry_price * (1.0 - (1.0 / leverage) + mmr) / (1.0 + taker_fee + maker_fee)
        )
    else:  # SELL (side_code == -1)
        # При продаже цена ликвидации выше
        return (
            entry_price * (1.0 + (1.0 / leverage) - mmr) / (1.0 - taker_fee - taker_fee)
        )


# Основное ядро симуляции, компилируемое Numba
@njit
def simulate_trades_numba_core(
    close_arr,
    high_arr,
    low_arr,
    timestamp_arr,  # Основные данные
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
    Возвращает список кортежей с результатами сделок.
    """
    current_equity = initial_capital

    # Структура текущей сделки: (Side, EntryPrice, SLPrice, TPPrice, PositionQty, LiquidationPrice, EquityAtEntry, EntryIndex)
    current_trade = None  # Используем None вместо словаря

    # (entry_index, exit_index, entry_price, exit_price, pnl, side_code, is_liquidation, close_reason_code, equity_after, position_qty)
    closed_trades = []

    # --- DEBUG COUNTERS ---
    long_signal_count = 0
    short_signal_count = 0
    # ----------------------

    # Переменные для отслеживания последней значимой волны
    last_swing_high_price = -1.0
    last_swing_low_price = -1.0
    last_swing_high_index = -1
    last_swing_low_index = -1

    data_len = len(close_arr)
    start_index = swing_window  # Начинаем после инициализации swing_window

    # Коды для причины закрытия:
    REASON_SL = 1
    REASON_TP = 2
    REASON_REVERSE_TREND = 3
    REASON_HTF_CHANGE = 4
    REASON_LIQUIDATION = 5
    REASON_END_OF_DATA = 6

    # --- ГЛАВНЫЙ ЦИКЛ БЭКТЕСТА ---
    for i in range(max(swing_window, start_index), data_len):
        current_time = timestamp_arr[i]
        current_high = high_arr[i]
        current_low = low_arr[i]
        current_close = close_arr[i]

        # Значения индикаторов для текущего бара (i)
        ema_fast = ema_fast_arr[i]
        ema_slow = ema_slow_arr[i]
        atr_val = atr_val_arr[i]
        htf_is_uptrend = htf_is_uptrend_arr[i]

        # --- 1. Обновление Структурных Точек ---

        # Обновляем Swing High
        if is_swing_high_arr[i]:
            if last_swing_low_index == -1 or i > last_swing_low_index:
                last_swing_high_price = high_arr[i]
                last_swing_high_index = i

        # Обновляем Swing Low
        if is_swing_low_arr[i]:
            if last_swing_high_index == -1 or i > last_swing_high_index:
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
            # Восходящее движение: ищем коррекцию
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                fibo_levels_list.append(last_swing_high_price - (diff * level))

        elif (
            is_short_wave_available
            and last_swing_high_price > 0
            and last_swing_low_price > 0
        ):
            # Нисходящее движение: ищем коррекцию
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                fibo_levels_list.append(last_swing_low_price + (diff * level))

        # --- 3. Условия Входа ---
        long_entry_cond = False
        short_entry_cond = False

        if is_uptrend and is_long_wave_available and htf_is_uptrend:
            long_signal_count += 1
            for fibo_level in fibo_levels_list:
                # Условие касания и отскока
                if (
                    fibo_level > current_low
                    and fibo_level < current_high
                    and current_close > fibo_level
                ):
                    long_entry_cond = True
                    break

        if is_downtrend and is_short_wave_available and not htf_is_uptrend:
            short_signal_count += 1
            for fibo_level in fibo_levels_list:
                # Условие касания и отскока
                if (
                    fibo_level > current_low
                    and fibo_level < current_high
                    and current_close < fibo_level
                ):
                    short_entry_cond = True
                    break

        # === Логика Закрытия Сделки ===
        # Используем явную проверку None и полную распаковку для Numba
        if current_trade is not None:
            # Распаковка кортежа для доступа к его элементам без ошибок Numba
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

            # 1. Проверка Стоп-Лосса
            if (side_code == 1 and current_low <= stop_loss_price) or (
                side_code == -1 and current_high >= stop_loss_price
            ):
                exit_price = stop_loss_price
                close_reason_code = REASON_SL

            # 2. Проверка Тейк-Профита
            elif (side_code == 1 and current_high >= take_profit_price) or (
                side_code == -1 and current_low <= take_profit_price
            ):
                exit_price = take_profit_price
                close_reason_code = REASON_TP

            # 3. Проверка обратного сигнала (выход по смене тренда)
            elif (side_code == 1 and is_downtrend) or (side_code == -1 and is_uptrend):
                exit_price = current_close
                close_reason_code = REASON_REVERSE_TREND

            # 4. Проверка на смену HTF тренда
            elif (side_code == 1 and not htf_is_uptrend) or (
                side_code == -1 and htf_is_uptrend
            ):
                exit_price = current_close
                close_reason_code = REASON_HTF_CHANGE

            # 5. Проверка ликвидации
            elif (side_code == 1 and current_low <= liquidation_price) or (
                side_code == -1 and current_high >= liquidation_price
            ):
                exit_price = liquidation_price
                close_reason_code = REASON_LIQUIDATION

            if close_reason_code != 0:
                # Расчет PNL
                qty = position_qty

                # Комиссии за вход и выход (maker + taker)
                fee_entry = qty * entry_price * (maker_fee + taker_fee)
                fee_exit = qty * exit_price * (maker_fee + taker_fee)

                if side_code == 1:  # BUY
                    pnl = (
                        (exit_price - entry_price) * qty * leverage
                        - fee_entry
                        - fee_exit
                    )
                else:  # SELL
                    pnl = (
                        (entry_price - exit_price) * qty * leverage
                        - fee_entry
                        - fee_exit
                    )

                current_equity += pnl
                is_liquidation = close_reason_code == REASON_LIQUIDATION

                # Сохраняем сделку как кортеж
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
        if current_trade is None:

            # Вычисляем необходимые ценовые движения в долларах
            sl_price_dist_usd = current_close * (
                required_price_move_for_sl_percent / 100.0
            )

            # Фильтр: SL должен быть больше, чем минимальная волатильность (ATR * множитель)
            min_sl_dist_by_atr = atr_val * atr_min_multiplier

            # Максимальная сумма, которую мы готовы потерять
            max_loss_usd = current_equity * (risk_per_trade_percent / 100.0)

            # Расчет Qty на основе риска
            if sl_price_dist_usd > 0.0:
                # Сколько Qty нужно, чтобы (SL_Dist * Qty * Leverage) = max_loss_usd
                risk_based_qty = max_loss_usd / (sl_price_dist_usd * leverage)
                margin_based_qty = (current_equity * leverage) / current_close
                position_qty = min(risk_based_qty, margin_based_qty)
            else:
                position_qty = 0.0

            # Проверки качества и комиссий
            quality_check_ok = sl_price_dist_usd >= min_sl_dist_by_atr
            estimated_total_commission = (
                position_qty * current_close * (maker_fee + taker_fee) * 2.0
            )
            commission_check_ok = estimated_total_commission < (max_loss_usd * 0.5)

            if position_qty > 0.0 and quality_check_ok and commission_check_ok:

                if long_entry_cond:
                    side_code = 1  # BUY
                    entry_price = current_close
                    lq_price = _calculate_liquidation_price_numba(
                        side_code, entry_price, leverage, maker_fee, taker_fee
                    )

                    # Цена SL и TP в долларах (РАССЧИТАНЫ ИЗ ROI%)
                    stop_loss_price = entry_price * (
                        1.0 - (required_price_move_for_sl_percent / 100.0)
                    )
                    take_profit_price = entry_price * (
                        1.0 + (required_price_move_for_tp_percent / 100.0)
                    )

                    # Сохраняем сделку как кортеж
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

                    # Цена SL и TP в долларах (РАССЧИТАНЫ ИЗ ROI%)
                    stop_loss_price = entry_price * (
                        1.0 + (required_price_move_for_sl_percent / 100.0)
                    )
                    take_profit_price = entry_price * (
                        1.0 - (required_price_move_for_tp_percent / 100.0)
                    )

                    # Сохраняем сделку как кортеж
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

    # Закрытие последней сделки, если она открыта (вне цикла Numba)
    if current_trade is not None:
        # Полная распаковка для доступа к элементам
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

        exit_price = close_arr[-1]
        exit_index = data_len - 1

        fee_entry = position_qty * entry_price * (maker_fee + taker_fee)
        fee_exit = position_qty * exit_price * (maker_fee + taker_fee)

        if side_code == 1:  # BUY
            pnl = (
                (exit_price - entry_price) * position_qty * leverage
                - fee_entry
                - fee_exit
            )
        else:  # SELL
            pnl = (
                (entry_price - exit_price) * position_qty * leverage
                - fee_entry
                - fee_exit
            )

        current_equity += pnl

        closed_trades.append(
            (
                entry_index,
                exit_index,
                entry_price,
                exit_price,
                pnl,
                side_code,
                False,
                REASON_END_OF_DATA,
                current_equity,
                position_qty,
            )
        )
        current_trade = None

    return closed_trades, long_signal_count, short_signal_count, current_equity


# === ЭТАП 3: Драйвер для Numba-симуляции ===

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


def simulate_trades_fss_roi(
    df,
    initial_capital,
    leverage,
    maker_fee,
    taker_fee,
    risk_per_trade_percent,
    required_price_move_for_sl_percent,
    required_price_move_for_tp_percent,
    fibo_levels=[0.5, 0.618],
    swing_window=500,
    atr_min_multiplier=1.5,
):
    """
    Драйвер для Numba-симуляции: Подготавливает данные и запускает ядро.
    """

    # Подготовка данных (преобразование в массивы NumPy)
    close_arr = df["close"].to_numpy(dtype=np.float64)
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr = df["low"].to_numpy(dtype=np.float64)
    timestamp_arr = df[
        "timestamp"
    ].to_numpy()  # Оставляем объектный тип для дальнейшего использования
    ema_fast_arr = df["ema_fast"].to_numpy(dtype=np.float64)
    ema_slow_arr = df["ema_slow"].to_numpy(dtype=np.float64)
    atr_val_arr = df["atr_val"].to_numpy(dtype=np.float64)
    htf_is_uptrend_arr = df["htf_is_uptrend"].to_numpy(dtype=np.bool_)
    is_swing_high_arr = df["is_swing_high"].to_numpy(dtype=np.bool_)
    is_swing_low_arr = df["is_swing_low"].to_numpy(dtype=np.bool_)

    # Запуск Numba-ядра
    closed_trades_tuples, long_signal_count, short_signal_count, final_equity = (
        simulate_trades_numba_core(
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
        )
    )

    # Преобразование Numba-результатов в Pandas DataFrame
    trades_array = np.array(closed_trades_tuples, dtype=TRADE_DTYPE)
    trades_df = pd.DataFrame(trades_array)

    if trades_df.empty:
        return trades_df

    # Обратное преобразование кодов в понятные значения
    trades_df["side"] = trades_df["side_code"].apply(
        lambda x: "BUY" if x == 1 else "SELL"
    )

    reason_map = {
        1: "Stop Loss",
        2: "Take Profit",
        3: "Reverse Trend",
        4: "HTF Trend Change",
        5: "Liquidation",
        6: "End of Data",
    }
    trades_df["close_reason"] = trades_df["close_reason_code"].map(reason_map)

    # Привязка индексов обратно к временным меткам
    trades_df["entry_time"] = df.loc[trades_df["entry_index"], "timestamp"].values
    trades_df["exit_time"] = df.loc[trades_df["exit_index"], "timestamp"].values

    # Удаление служебных колонок
    trades_df = trades_df.drop(
        columns=["side_code", "close_reason_code", "entry_index", "exit_index"]
    )

    # --- DEBUG PRINT ---
    print(f"\n--- DEBUG: Обнаружено потенциальных LONG сигналов: {long_signal_count}")
    print(f"--- DEBUG: Обнаружено потенциальных SHORT сигналов: {short_signal_count}")
    # -------------------

    return trades_df


# === ЭТАП 4: Анализ Результатов (Консольный Вывод - без изменений) ===
def analyze_results(trades_df, initial_capital):
    """Рассчитывает и отображает ключевые метрики бэктеста в виде консольной таблицы."""
    if trades_df.empty:
        print("\nНет закрытых сделок для анализа.")
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)

    # 1. Расчет основных метрик
    total_pnl = trades_df["pnl"].sum()
    success_rate = (trades_df["pnl"] > 0).mean() * 100
    liquidation_rate = trades_df["liquidation"].mean() * 100
    num_trades = len(trades_df)

    final_equity = trades_df["equity_after_trade"].iloc[-1]
    return_on_capital = (final_equity - initial_capital) / initial_capital * 100

    # 2. Расчет Максимальной Просадки (Max Drawdown - MDD)
    equity_curve = trades_df["equity_after_trade"]
    peak = equity_curve.expanding().max()
    drawdown = (peak - equity_curve) / peak
    max_drawdown = drawdown.max() * 100 if not drawdown.empty else 0.0

    # 3. Подготовка данных для консольной таблицы
    metrics = {
        "Начальный капитал": f"${initial_capital:,.2f}",
        "Финальный капитал": f"${final_equity:,.2f}",
        "Общий PNL": f"${total_pnl:,.2f}",
        "ROI (Прибыльность)": f"{return_on_capital:.2f}%",
        "Максимальная просадка (MDD)": f"{max_drawdown:.2f}%",
        "Общее количество сделок": f"{num_trades}",
        "Процент прибыльных сделок": f"{success_rate:.1f}%",
        "Процент ликвидаций": f"{liquidation_rate:.1f}%",
    }

    # 4. Вывод Итоговой Сводки
    print("\n" + "=" * 60)
    print(" " * 10 + "РЕЗУЛЬТАТЫ FIBO STRUCTURE SCALPER (ROI-Driven)")
    print("=" * 60)

    max_key_len = max(len(k) for k in metrics.keys())

    for key, value in metrics.items():
        print(f"| {key:<{max_key_len}} | {value:>25} |")

    print("=" * 60)

    # 5. Вывод Лога Закрытых Сделок
    print("\n--- ЛОГ ЗАКРЫТЫХ СДЕЛОК (Первые/Последние) ---")
    log_cols = [
        "entry_time",
        "side",
        "entry_price",
        "exit_price",
        "pnl",
        "close_reason",
        "equity_after_trade",
    ]

    if len(trades_df) > 10:
        log_df = pd.concat([trades_df[log_cols].head(5), trades_df[log_cols].tail(5)])
        print(f"Показано 5 первых и 5 последних из {len(trades_df)} сделок:")
    else:
        log_df = trades_df[log_cols]

    print(log_df.to_string())
    print("-" * 60)

    return drawdown, equity_curve  # Возвращаем данные для построения графиков


# === ЭТАП 5: Построение графиков (Без изменений) ===
def plot_results(drawdown_data, equity_curve, initial_capital):
    """Строит графики кривой эквити и максимальной просадки."""

    if equity_curve.empty:
        print("Недостаточно данных для построения графиков.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- График 1: Кривая Эквити ---
    axes[0].plot(equity_curve.index, equity_curve, label="Кривая Эквити", color="green")
    axes[0].axhline(
        initial_capital, color="grey", linestyle="--", label="Начальный Капитал"
    )
    axes[0].set_title("1. Кривая Эквити (Equity Curve)", fontsize=14)
    axes[0].set_xlabel("Номер Сделки")
    axes[0].set_ylabel("Капитал ($)")
    axes[0].grid(True, alpha=0.5)
    axes[0].legend()

    # --- График 2: Максимальная Просадка ---
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


# === ОСНОВНОЙ СКРИПТ ===
if __name__ == "__main__":

    # =========================================================================
    # --- НАСТРОЙКА ПАРАМЕТРОВ СТРАТЕГИИ ---
    # =========================================================================

    # --- 1. Финансовые Параметры и Риск ---

    initial_capital = 100.0  # Начальный размер капитала.
    leverage = 10.0  # Плечо (Leverage).

    # Целевой ROI (Процент прибыли от общего капитала при успешной сделке).
    target_roi_percent = 20.0

    # Риск ROI (Процент потери от общего капитала при срабатывании SL).
    risk_roi_percent = 10.0

    maker_fee = 0.0002  # Комиссия мейкера.
    taker_fee = 0.0005  # Комиссия тейкера.

    # --- 2. Параметры Индикаторов (1-минутный ТФ) ---

    ema_fast_len = 9
    ema_slow_len = 21
    swing_window = 500
    atr_exit_len = 14
    atr_min_multiplier = 1.5

    # --- 3. Параметры Старшего Таймфрейма (HTF Filter) ---

    htf_period = (
        "15min"  # Высокий таймфрейм (15 минут). Допустимо: '5min', '1H', '4H' и т.д.
    )
    htf_ema_fast_len = 9  # Быстрая EMA для HTF
    htf_ema_slow_len = 21  # Медленная EMA для HTF
    fibo_levels = np.array(
        [0.5, 0.618], dtype=np.float64
    )  # Должен быть np.array для Numba

    # =========================================================================
    # --- РАСЧЕТ ПРОИЗВОДНЫХ ПАРАМЕТРОВ (Автоматически) ---
    # =========================================================================

    # Процент движения цены, необходимый для достижения Target ROI:
    required_price_move_for_tp_percent = target_roi_percent / leverage

    # Процент движения цены, необходимый для достижения Risk ROI:
    required_price_move_for_sl_percent = risk_roi_percent / leverage

    # Максимальный риск на сделку в долларах (равен Risk ROI в данном случае)
    risk_per_trade_percent = risk_roi_percent

    print("--- Автоматический Расчет Параметров ---")
    print(f"Целевое движение цены для TP: {required_price_move_for_tp_percent:.2f}%")
    print(
        f"Макс. допустимое движение цены для SL: {required_price_move_for_sl_percent:.2f}%"
    )
    print("-" * 45 + "\n")

    # =========================================================================
    # --- ЗАПУСК БЭКТЕСТА ---
    # =========================================================================

    # --- Загрузка и расчет 1m данных ---
    df = load_and_analyze_data(file_path)
    df = calculate_fibo_structure_indicators(
        df.copy(), ema_fast_len, ema_slow_len, atr_exit_len, swing_window
    )

    # --- Расчет HTF EMAs и их слияние ---

    # 1. Создаем HTF DF
    df_htf = (
        df.set_index("timestamp")
        .resample(htf_period)
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

    # 2. Рассчитываем HTF EMAs
    df_htf["htf_ema_fast"] = talib.EMA(df_htf["close"], timeperiod=htf_ema_fast_len)
    df_htf["htf_ema_slow"] = talib.EMA(df_htf["close"], timeperiod=htf_ema_slow_len)

    # Определяем булевый тренд HTF
    df_htf["htf_is_uptrend"] = df_htf["htf_ema_fast"] > df_htf["htf_ema_slow"]

    # 3. Добавляем HTF тренд обратно в 1m DF
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

    # Убеждаемся, что мы убрали NaN, которые появились в начале из-за HTF EMA
    df = df.dropna().reset_index(drop=True)

    print(f"--- DEBUG: После слияния с HTF осталось {len(df)} строк.")

    # --- Симуляция сделок (ВЫЗЫВАЕТСЯ Numba-ЯДРО) ---
    # Numba-функция компилируется при первом вызове, что может занять секунду.
    import time

    start_time = time.time()

    trades_df = simulate_trades_fss_roi(
        df,
        initial_capital,
        leverage,
        maker_fee,
        taker_fee,
        risk_per_trade_percent,
        required_price_move_for_sl_percent,
        required_price_move_for_tp_percent,
        fibo_levels=fibo_levels,
        swing_window=swing_window,
        atr_min_multiplier=atr_min_multiplier,
    )

    end_time = time.time()
    print(
        f"\n[TIME] Время выполнения симуляции (Numba): {end_time - start_time:.4f} секунд."
    )

    # --- Анализ результатов (консоль) ---
    drawdown_data, equity_curve_data = analyze_results(trades_df, initial_capital)

    # --- Построение графиков ---
    plot_results(drawdown_data, equity_curve_data, initial_capital)
