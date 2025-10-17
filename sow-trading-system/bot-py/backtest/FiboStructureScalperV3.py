import os
import pandas as pd
import talib
import numpy as np

# Импортируем библиотеку для графиков
import matplotlib.pyplot as plt
from math import copysign

# Определяем путь к файлу
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    script_dir,
    "BTC_USDT_1m_2023-08-01 00_00_00_to_2025-08-01_00_00_00.csv",
)

if not os.path.exists(file_path):
    print(f"Ошибка: Файл {file_path} не найден!")
    print("Убедитесь, что файл находится в той же папке, что и скрипт.")
    exit()


# === ЭТАП 1: Загрузка и анализ данных ===
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


# === ЭТАП 2: Расчет Индикаторов и Структурных Точек (FIBO/SWING) ===
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

    # --- DEBUG PRINT ---
    print(f"--- DEBUG: После расчета индикаторов осталось {len(df)} строк.")
    # -------------------

    return df


# Вспомогательная функция для расчета цены ликвидации
def calculate_liquidation_price(side, entry_price, leverage, maker_fee, taker_fee):
    """Рассчитывает приблизительную цену ликвидации."""
    mmr = 0.006
    if side == "BUY":
        # При покупке цена ликвидации ниже
        return entry_price * (1 - (1 / leverage) + mmr) / (1 + taker_fee + maker_fee)
    else:  # SELL
        # При продаже цена ликвидации выше
        return entry_price * (1 + (1 / leverage) - mmr) / (1 - taker_fee - taker_fee)


# === ЭТАП 3: Симуляция Сделок по Fibo Structure Scalper (FSS) ===
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
    atr_min_multiplier=1.5,  # Минимальный ATR множитель для принятия сделки (фильтр)
):
    """Симуляция торговли по стратегии Fibo Structure Scalper с ROI-Driven логикой и HTF фильтром."""
    current_equity = initial_capital
    current_trade = None
    closed_trades = []

    # --- DEBUG COUNTERS ---
    long_signal_count = 0
    short_signal_count = 0
    trades_filtered_by_atr = 0
    # ----------------------

    # Переменные для отслеживания последней значимой волны
    last_swing_high_price = None
    last_swing_low_price = None
    last_swing_high_index = -1
    last_swing_low_index = -1

    # Индекс, с которого начинаем анализ (чтобы убедиться, что есть данные HTF)
    start_index = df["htf_is_uptrend"].first_valid_index()
    if start_index is None:
        print("Ошибка: Нет данных HTF для анализа. Проверьте настройки таймфреймов.")
        return pd.DataFrame(closed_trades)

    for i in range(max(swing_window, start_index), len(df)):
        current_time = df.iloc[i]["timestamp"]
        current_high = df.iloc[i]["high"]
        current_low = df.iloc[i]["low"]
        current_close = df.iloc[i]["close"]

        # Значения индикаторов для текущего бара (i)
        ema_fast = df.iloc[i]["ema_fast"]
        ema_slow = df.iloc[i]["ema_slow"]
        atr_val = df.iloc[i]["atr_val"]
        htf_is_uptrend = df.iloc[i]["htf_is_uptrend"]  # Получаем HTF тренд

        # --- 1. Обновление Структурных Точек (Ищем в истории) ---

        # Обновляем Swing High
        if df.iloc[i]["is_swing_high"]:
            if last_swing_low_index == -1 or i > last_swing_low_index:
                last_swing_high_price = df.iloc[i]["high"]
                last_swing_high_index = i

        # Обновляем Swing Low
        if df.iloc[i]["is_swing_low"]:
            if last_swing_high_index == -1 or i > last_swing_high_index:
                last_swing_low_price = df.iloc[i]["low"]
                last_swing_low_index = i

        # --- 2. Расчет Уровней Фибоначчи ---
        fibo_levels_dict = {}

        is_uptrend = ema_fast > ema_slow
        is_downtrend = ema_fast < ema_slow

        # Есть ли полная волна (Low -> High для LONG, High -> Low для SHORT)
        is_long_wave_available = (
            last_swing_high_index > last_swing_low_index and last_swing_low_index != -1
        )
        is_short_wave_available = (
            last_swing_low_index > last_swing_high_index and last_swing_high_index != -1
        )

        if (
            is_long_wave_available
            and last_swing_high_price is not None
            and last_swing_low_price is not None
        ):
            # Восходящее движение: ищем коррекцию
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                fibo_levels_dict[level] = last_swing_high_price - (diff * level)

        elif (
            is_short_wave_available
            and last_swing_high_price is not None
            and last_swing_low_price is not None
        ):
            # Нисходящее движение: ищем коррекцию
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                fibo_levels_dict[level] = last_swing_low_price + (diff * level)

        # --- 3. Условия Входа (Тренд 1m + HTF Фильтр + Касание Уровня + Отскок) ---

        long_entry_cond = False
        short_entry_cond = False

        # LONG: Тренд 1m UP + Тренд HTF UP
        if is_uptrend and is_long_wave_available and htf_is_uptrend:
            long_signal_count += 1
            for fibo_level in fibo_levels_dict.values():
                # Условие касания: Свеча проколола уровень (Low <= Fibo <= High)
                # Условие отскока: Закрылась выше Fibo
                if (
                    fibo_level > current_low
                    and fibo_level < current_high
                    and current_close > fibo_level
                ):
                    long_entry_cond = True
                    break

        # SHORT: Тренд 1m DOWN + Тренд HTF DOWN
        if is_downtrend and is_short_wave_available and not htf_is_uptrend:
            short_signal_count += 1
            for fibo_level in fibo_levels_dict.values():
                # Условие касания: Свеча проколола уровень (Low <= Fibo <= High)
                # Условие отскока: Закрылась ниже Fibo
                if (
                    fibo_level > current_low
                    and fibo_level < current_high
                    and current_close < fibo_level
                ):
                    short_entry_cond = True
                    break

        # === Логика Закрытия Сделки ===
        if current_trade:
            side = current_trade["side"]
            entry_price = current_trade["entry_price"]
            exit_price = None
            close_reason = None

            # SL/TP Price (были зафиксированы при входе)
            stop_loss_price = current_trade["stop_loss_price"]
            take_profit_price = current_trade["take_profit_price"]

            # 1. Проверка Стоп-Лосса
            if (side == "BUY" and current_low <= stop_loss_price) or (
                side == "SELL" and current_high >= stop_loss_price
            ):
                exit_price = stop_loss_price
                close_reason = "Stop Loss"

            # 2. Проверка Тейк-Профита
            if close_reason is None:
                if (side == "BUY" and current_high >= take_profit_price) or (
                    side == "SELL" and current_low <= take_profit_price
                ):
                    exit_price = take_profit_price
                    close_reason = "Take Profit"

            # 3. Проверка обратного сигнала (выход по смене тренда)
            if close_reason is None:
                if (side == "BUY" and is_downtrend) or (side == "SELL" and is_uptrend):
                    exit_price = current_close
                    close_reason = "Reverse Trend"

            # 4. Проверка на смену HTF тренда (Дополнительный выход для безопасности)
            if close_reason is None:
                if (side == "BUY" and not htf_is_uptrend) or (
                    side == "SELL" and htf_is_uptrend
                ):
                    exit_price = current_close
                    close_reason = "HTF Trend Change"

            # 5. Проверка ликвидации (крайний случай)
            lq_price = current_trade["liquidation_price"]
            if close_reason is None:
                if (side == "BUY" and current_low <= lq_price) or (
                    side == "SELL" and current_high >= lq_price
                ):
                    exit_price = lq_price
                    close_reason = "Liquidation"

            if close_reason is not None:
                # Расчет PNL
                qty = current_trade["position_qty"]

                # Комиссии за вход и выход (maker + taker)
                fee_entry = qty * entry_price * (maker_fee + taker_fee)
                fee_exit = qty * exit_price * (maker_fee + taker_fee)

                if side == "BUY":
                    pnl = (
                        (exit_price - entry_price) * qty * leverage
                        - fee_entry
                        - fee_exit
                    )
                else:
                    pnl = (
                        (entry_price - exit_price) * qty * leverage
                        - fee_entry
                        - fee_exit
                    )

                current_equity += pnl

                current_trade.update(
                    {
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "success": pnl > 0,
                        "liquidation": (close_reason == "Liquidation"),
                        "close_reason": close_reason,
                        "equity_after_trade": current_equity,
                    }
                )
                closed_trades.append(current_trade)
                current_trade = None

        # === Логика Открытия Сделки ===
        if not current_trade:

            # Вычисляем необходимые ценовые движения в долларах
            sl_price_dist_usd = current_close * (
                required_price_move_for_sl_percent / 100
            )

            # Фильтр: SL должен быть больше, чем минимальная волатильность (ATR * множитель)
            min_sl_dist_by_atr = atr_val * atr_min_multiplier

            # --- Динамический расчет размера позиции на основе риска ---

            # Максимальная сумма, которую мы готовы потерять (ROI Risk % от капитала)
            max_loss_usd = current_equity * (risk_per_trade_percent / 100)

            # Объем позиции (Qty) рассчитывается, чтобы потерять max_loss_usd при достижении SL
            if sl_price_dist_usd > 0:
                # Сколько Qty нужно, чтобы (SL_Dist * Qty * Leverage) = max_loss_usd
                risk_based_qty = max_loss_usd / (sl_price_dist_usd * leverage)
                margin_based_qty = (current_equity * leverage) / current_close
                position_qty = min(risk_based_qty, margin_based_qty)
            else:
                position_qty = 0.0

            # Проверка, что SL достаточно широкий по сравнению с ATR (фильтр качества)
            quality_check_ok = sl_price_dist_usd >= min_sl_dist_by_atr

            # Проверка, что сделка имеет смысл (комиссия не съедает большую часть риска)
            estimated_total_commission = (
                position_qty * current_close * (maker_fee + taker_fee) * 2
            )
            commission_check_ok = estimated_total_commission < (max_loss_usd * 0.5)

            if position_qty > 0 and quality_check_ok and commission_check_ok:

                if long_entry_cond:
                    side = "BUY"
                    entry_price = current_close
                    lq_price = calculate_liquidation_price(
                        side, entry_price, leverage, maker_fee, taker_fee
                    )

                    # Цена SL и TP в долларах (РАССЧИТАНЫ ИЗ ROI%)
                    stop_loss_price = entry_price * (
                        1 - (required_price_move_for_sl_percent / 100)
                    )
                    take_profit_price = entry_price * (
                        1 + (required_price_move_for_tp_percent / 100)
                    )

                    current_trade = {
                        "entry_time": current_time,
                        "entry_price": entry_price,
                        "side": side,
                        "liquidation_price": lq_price,
                        "position_qty": position_qty,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "equity_at_entry": current_equity,
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": None,
                        "success": None,
                        "liquidation": False,
                        "close_reason": None,
                    }

                elif short_entry_cond:
                    side = "SELL"
                    entry_price = current_close
                    lq_price = calculate_liquidation_price(
                        side, entry_price, leverage, maker_fee, taker_fee
                    )

                    # Цена SL и TP в долларах (РАССЧИТАНЫ ИЗ ROI%)
                    stop_loss_price = entry_price * (
                        1 + (required_price_move_for_sl_percent / 100)
                    )
                    take_profit_price = entry_price * (
                        1 - (required_price_move_for_tp_percent / 100)
                    )

                    current_trade = {
                        "entry_time": current_time,
                        "entry_price": entry_price,
                        "side": side,
                        "liquidation_price": lq_price,
                        "position_qty": position_qty,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "equity_at_entry": current_equity,
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": None,
                        "success": None,
                        "liquidation": False,
                        "close_reason": None,
                    }

    # Закрытие последней сделки, если она открыта
    if current_trade:
        side = current_trade["side"]
        entry_price = current_trade["entry_price"]
        exit_price = df.iloc[-1]["close"]

        fee_entry = (
            current_trade["position_qty"] * entry_price * (maker_fee + taker_fee)
        )
        fee_exit = current_trade["position_qty"] * exit_price * (maker_fee + taker_fee)

        if side == "BUY":
            pnl = (
                (exit_price - entry_price) * current_trade["position_qty"] * leverage
                - fee_entry
                - fee_exit
            )
        else:
            pnl = (
                (entry_price - exit_price) * current_trade["position_qty"] * leverage
                - fee_entry
                - fee_exit
            )

        current_equity += pnl

        current_trade.update(
            {
                "exit_time": df.iloc[-1]["timestamp"],
                "exit_price": exit_price,
                "pnl": pnl,
                "success": pnl > 0,
                "liquidation": False,
                "close_reason": "End of Data",
                "equity_after_trade": current_equity,
            }
        )
        closed_trades.append(current_trade)

    # --- DEBUG PRINT ---
    print(f"\n--- DEBUG: Обнаружено потенциальных LONG сигналов: {long_signal_count}")
    print(f"--- DEBUG: Обнаружено потенциальных SHORT сигналов: {short_signal_count}")
    print(f"--- DEBUG: Отфильтровано по качеству ATR: {trades_filtered_by_atr}")
    # -------------------

    return pd.DataFrame(closed_trades)


# === ЭТАП 4: Анализ Результатов (Консольный Вывод) ===
def analyze_results(trades_df, initial_capital):
    """Рассчитывает и отображает ключевые метрики бэктеста в виде консольной таблицы."""
    if trades_df.empty:
        print("\nНет закрытых сделок для анализа.")
        return

    # 1. Расчет основных метрик
    total_pnl = trades_df["pnl"].sum()
    success_rate = (trades_df["pnl"] > 0).mean() * 100
    liquidation_rate = trades_df["liquidation"].mean() * 100
    num_trades = len(trades_df)

    final_equity = initial_capital + total_pnl
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
    # --- НАСТРОЙКА ПАРАМЕТРОВ СТРАТЕГИИ (Централизованно) ---
    # =========================================================================

    # --- 1. Финансовые Параметры и Риск (Ввод Пользователя) ---

    initial_capital = 100.0  # Начальный размер капитала.
    leverage = 20.0  # Плечо (Leverage).

    # Целевой ROI (Процент прибыли от общего капитала при успешной сделке).
    target_roi_percent = 18.0

    # Риск ROI (Процент потери от общего капитала при срабатывании SL).
    risk_roi_percent = 10.0

    # (20 | 15 + 10 = PNL $12.95)
    # (20 | 17 + 10 = PNL $116.91)
    # (20 | 18 + 10 = PNL $118.92)

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
    # Используем 'close' как метку времени для завершения периода
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
    df_htf["htf_ema_fast"] = talib.EMA(
        df_htf["close"].to_numpy(dtype=float), timeperiod=htf_ema_fast_len
    )
    df_htf["htf_ema_slow"] = talib.EMA(
        df_htf["close"].to_numpy(dtype=float), timeperiod=htf_ema_slow_len
    )

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

    # --- Симуляция сделок ---
    trades_df = simulate_trades_fss_roi(
        df,
        initial_capital,
        leverage,
        maker_fee,
        taker_fee,
        risk_per_trade_percent,
        required_price_move_for_sl_percent,
        required_price_move_for_tp_percent,
        swing_window=swing_window,
        atr_min_multiplier=atr_min_multiplier,
    )

    # --- Анализ результатов (консоль) ---
    analyze_result = analyze_results(trades_df, initial_capital)
    if analyze_result is not None:
        drawdown_data, equity_curve_data = analyze_result
        # --- Построение графиков ---
        plot_results(drawdown_data, equity_curve_data, initial_capital)
    else:
        print("Недостаточно данных для построения графиков.")
