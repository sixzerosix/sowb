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
    "BTC_USDT_1m_2025-07-01_00_00_00_to_2025-09-01_00_00_00.csv",
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
    """Расчет индикаторов и ключевых структурных точек для стратегии FSS."""

    # 1. EMAs (9 и 21) для определения тренда
    df["ema_fast"] = talib.EMA(df["close"], timeperiod=ema_fast_len)
    df["ema_slow"] = talib.EMA(df["close"], timeperiod=ema_slow_len)

    # 2. ATR для стопов
    df["atr_exit"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=atr_len)

    # 3. Расчет структурных точек (Swing High/Low)
    # Используем Rolling Window для поиска максимумов и минимумов в окне (например, 100 баров)
    df["is_swing_high"] = (
        df["high"] == df["high"].rolling(window=swing_window, center=True).max()
    )
    df["is_swing_low"] = (
        df["low"] == df["low"].rolling(window=swing_window, center=True).min()
    )

    # Заполняем NaN, которые возникли из-за расчетов
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
        return entry_price * (1 - (1 / leverage) + mmr) / (1 + taker_fee + maker_fee)
    else:  # SELL
        return entry_price * (1 + (1 / leverage) - mmr) / (1 - taker_fee - maker_fee)


# === ЭТАП 3: Симуляция Сделок по Fibo Structure Scalper (FSS) ===
def simulate_trades_fss(
    df,
    initial_capital,
    leverage,
    maker_fee,
    taker_fee,
    risk_per_trade_percent,
    stop_loss_atr_mult,
    take_profit_atr_mult,
    fibo_levels=[0.5, 0.618],  # Ключевые уровни коррекции
    lookback_bars=500,  # Окно для поиска Swing точек
):
    """Симуляция торговли по стратегии Fibo Structure Scalper (FSS)."""
    current_equity = initial_capital
    current_trade = None
    closed_trades = []

    # --- DEBUG COUNTERS ---
    long_signal_count = 0
    short_signal_count = 0
    # ----------------------

    # Переменные для отслеживания последней значимой волны
    last_swing_high_price = None
    last_swing_low_price = None
    last_swing_high_index = -1
    last_swing_low_index = -1

    for i in range(1, len(df)):
        current_time = df.iloc[i]["timestamp"]
        current_high = df.iloc[i]["high"]
        current_low = df.iloc[i]["low"]
        current_close = df.iloc[i]["close"]

        # Значения индикаторов для текущего бара (i)
        ema_fast = df.iloc[i]["ema_fast"]
        ema_slow = df.iloc[i]["ema_slow"]
        atr_exit = df.iloc[i]["atr_exit"]

        # --- 1. Обновление Структурных Точек ---

        # Обновляем Swing High, если он найден и находится в окне lookback
        if df.iloc[i]["is_swing_high"] and i - last_swing_low_index > 0:
            last_swing_high_price = df.iloc[i]["high"]
            last_swing_high_index = i

        # Обновляем Swing Low, если он найден и находится в окне lookback
        if df.iloc[i]["is_swing_low"] and i - last_swing_high_index > 0:
            last_swing_low_price = df.iloc[i]["low"]
            last_swing_low_index = i

        # --- 2. Расчет Уровней Фибоначчи (только если есть последняя волна) ---
        fibo_levels_dict = {}

        # Восходящий тренд (движение вверх) - Коррекция вниз, ищем LONG
        is_uptrend = ema_fast > ema_slow
        # Нисходящий тренд (движение вниз) - Коррекция вверх, ищем SHORT
        is_downtrend = ema_fast < ema_slow

        is_uptrend_wave = (
            last_swing_high_index > last_swing_low_index and last_swing_low_index != -1
        )
        is_downtrend_wave = (
            last_swing_low_index > last_swing_high_index and last_swing_high_index != -1
        )

        if (
            is_uptrend_wave
            and last_swing_high_price is not None
            and last_swing_low_price is not None
        ):
            # Расчет коррекции для восходящего движения
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                fibo_levels_dict[level] = last_swing_high_price - (diff * level)

        elif (
            is_downtrend_wave
            and last_swing_high_price is not None
            and last_swing_low_price is not None
        ):
            # Расчет коррекции для нисходящего движения
            diff = last_swing_high_price - last_swing_low_price
            for level in fibo_levels:
                fibo_levels_dict[level] = last_swing_low_price + (diff * level)

        # --- 3. Условия Входа (Конфлюенция: Тренд + Касание Уровня) ---

        long_entry_cond = False
        short_entry_cond = False

        # Проверка касания уровня Фибо для LONG
        if is_uptrend and is_uptrend_wave:
            for fibo_level in fibo_levels_dict.values():
                # Проверяем, что цена закрылась между уровнем и Swing Low
                # (То есть, мы корректировались к уровню и отскочили от него)
                if (
                    fibo_level > current_low
                    and fibo_level < current_high
                    and current_close > fibo_level
                ):
                    long_entry_cond = True
                    break

        # Проверка касания уровня Фибо для SHORT
        if is_downtrend and is_downtrend_wave:
            for fibo_level in fibo_levels_dict.values():
                # Проверяем, что цена закрылась между уровнем и Swing High
                # (То есть, мы корректировались к уровню и отскочили от него)
                if (
                    fibo_level < current_high
                    and fibo_level > current_low
                    and current_close < fibo_level
                ):
                    short_entry_cond = True
                    break

        # --- DEBUG COUNTERS UPDATE ---
        if long_entry_cond:
            long_signal_count += 1
        if short_entry_cond:
            short_signal_count += 1
        # -----------------------------

        # === Логика Закрытия Сделки (Без изменений) ===
        if current_trade:
            # ... (логика закрытия остается такой же, как в предыдущей версии) ...
            side = current_trade["side"]
            entry_price = current_trade["entry_price"]
            entry_atr_at_open = current_trade["entry_atr"]
            exit_price = None
            close_reason = None

            # Расчет SL/TP на основе ATR, зафиксированного при входе
            stop_loss_dist = stop_loss_atr_mult * entry_atr_at_open
            take_profit_dist = take_profit_atr_mult * entry_atr_at_open

            if side == "BUY":
                stop_loss_price = entry_price - stop_loss_dist
                take_profit_price = entry_price + take_profit_dist
            else:  # SELL
                stop_loss_price = entry_price + stop_loss_dist
                take_profit_price = entry_price - take_profit_dist

            # 1. Проверка Стоп-Лосса
            if (side == "BUY" and current_low <= stop_loss_price) or (
                side == "SELL" and current_high >= stop_loss_price
            ):
                exit_price = stop_loss_price
                close_reason = "Stop Loss"

            # 2. Проверка Тейк-Профита (используем только если он задан)
            if close_reason is None and take_profit_atr_mult > 0:
                if (side == "BUY" and current_high >= take_profit_price) or (
                    side == "SELL" and current_low <= take_profit_price
                ):
                    exit_price = take_profit_price
                    close_reason = "Take Profit"

            # 3. Проверка обратного сигнала (выход по смене тренда)
            if close_reason is None:
                # В стратегии FSS выход по обратному сигналу упрощен до смены тренда
                if (side == "BUY" and is_downtrend) or (side == "SELL" and is_uptrend):
                    exit_price = current_close
                    close_reason = "Reverse Trend"

            # 4. Проверка ликвидации (крайний случай)
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

        # === Логика Открытия Сделки (Используем новые условия) ===
        if not current_trade:
            # Динамический расчет размера позиции на основе риска
            position_qty = 0.0
            max_loss_usd = current_equity * (risk_per_trade_percent / 100)

            # Расстояние SL в абсолютных долларах
            calculated_stop_loss_dist_usd = atr_exit * stop_loss_atr_mult

            if calculated_stop_loss_dist_usd > 0:
                risk_based_qty = max_loss_usd / calculated_stop_loss_dist_usd
                margin_based_qty = (current_equity * leverage) / current_close
                position_qty = min(risk_based_qty, margin_based_qty)

            # Проверка, что сделка имеет смысл (комиссия)
            estimated_total_commission = (
                position_qty * current_close * (maker_fee + taker_fee) * 2
            )
            # Комиссия должна быть меньше 50% от максимального риска на сделку
            commission_check_ok = estimated_total_commission < (max_loss_usd * 0.5)

            if position_qty > 0 and commission_check_ok:

                if long_entry_cond:
                    side = "BUY"
                    entry_price = current_close
                    lq_price = calculate_liquidation_price(
                        side, entry_price, leverage, maker_fee, taker_fee
                    )

                    current_trade = {
                        "entry_time": current_time,
                        "entry_price": entry_price,
                        "side": side,
                        "liquidation_price": lq_price,
                        "position_qty": position_qty,
                        "entry_atr": atr_exit,
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

                    current_trade = {
                        "entry_time": current_time,
                        "entry_price": entry_price,
                        "side": side,
                        "liquidation_price": lq_price,
                        "position_qty": position_qty,
                        "entry_atr": atr_exit,
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
    print(" " * 10 + "РЕЗУЛЬТАТЫ FIBO STRUCTURE SCALPER (FSS)")
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


# === ЭТАП 5: Построение графиков ===
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

    # --- 1. Параметры Финансирования и Риска ---

    initial_capital = 100.0
    leverage = 10.0
    maker_fee = 0.0002
    taker_fee = 0.0005
    risk_per_trade_percent = 2.0

    # --- 2. Параметры Индикаторов (FIBO STRUCTURE) ---

    # EMAs для определения тренда
    ema_fast_len = 9
    ema_slow_len = 21

    # Окно поиска Swing High/Low. Чем больше окно, тем более крупную волну мы ищем.
    # 100 баров на 1м графике = 1 час истории.
    swing_window = 100

    # --- 3. Параметры Выхода (ATR Stops) ---

    atr_exit_len = 14

    # Стоп-Лосс: 1.5 * ATR (Агрессивный скальпинг)
    stop_loss_atr_mult = 1.5

    # Тейк-Профит: 3.0 * ATR (Соотношение Risk/Reward 1:2)
    take_profit_atr_mult = 3.0

    # =========================================================================
    # --- ЗАПУСК БЭКТЕСТА ---
    # =========================================================================

    # --- Загрузка и расчет данных ---
    df = load_and_analyze_data(file_path)
    df = calculate_fibo_structure_indicators(
        df.copy(), ema_fast_len, ema_slow_len, atr_exit_len, swing_window
    )

    # --- Симуляция сделок ---
    trades_df = simulate_trades_fss(
        df,
        initial_capital,
        leverage,
        maker_fee,
        taker_fee,
        risk_per_trade_percent,
        stop_loss_atr_mult,
        take_profit_atr_mult,
    )

    # --- Анализ результатов (консоль) ---
    analyze_result = analyze_results(trades_df, initial_capital)
    if analyze_result is not None:
        drawdown_data, equity_curve_data = analyze_result
        # --- Построение графиков ---
        plot_results(drawdown_data, equity_curve_data, initial_capital)
    else:
        print("Недостаточно данных для построения графиков.")
