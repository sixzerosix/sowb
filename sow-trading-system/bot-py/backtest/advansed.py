import os
import pandas as pd
import talib
import matplotlib.pyplot as plt
import numpy as np

# Определяем путь к файлу
script_dir = os.path.dirname(os.path.abspath(__file__))
# Используем os.path.join для создания кроссплатформенного пути
file_path = os.path.join(
    script_dir,
    "BTC_USDT_1m_2025-07-01_00_00_00_to_2025-09-01_00_00_00.csv",
)

# Проверяем, существует ли файл
if not os.path.exists(file_path):
    print(f"Ошибка: Файл {file_path} не найден!")
    print("Убедитесь, что файл находится в той же папке, что и скрипт.")
    exit()


# Этап 1: Загрузка и анализ данных
def load_and_analyze_data(file_path):
    """Загрузка, сортировка и базовый анализ данных."""
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
    except Exception as e:
        print(f"Ошибка при чтении файла CSV: {e}")
        exit()

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Первые 5 строк:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())
    print("\nОписательная статистика:")
    print(df.describe())

    # Визуализация (отключена для автоматического выполнения, раскомментируйте для просмотра)
    # plt.figure(figsize=(14, 7))
    # plt.plot(df["timestamp"], df["close"], label="Цена закрытия", linewidth=1)
    # plt.title("График цены BTC/USDT (1 минута)")
    # plt.xlabel("Время")
    # plt.ylabel("Цена (USDT)")
    # plt.legend()
    # plt.grid()
    plt.show()

    return df


# Этап 2: Расчет всех индикаторов и условий для стратегии
def calculate_all_strategy_components(
    df,
    ema_fast_len,
    ema_slow_len,
    rsi_len,
    obv_ma_len,
    atr_len_vol,
    htf_ema_len,
    atr_exit_len,
):
    """Расчет всех необходимых технических индикаторов."""

    # EMA
    df["ema_fast"] = talib.EMA(df["close"], timeperiod=ema_fast_len)
    df["ema_slow"] = talib.EMA(df["close"], timeperiod=ema_slow_len)

    # VWAP (Прокси: используем длинную EMA)
    df["vwap_val"] = talib.EMA(df["close"], timeperiod=50)

    # RSI
    df["rsi_val"] = talib.RSI(df["close"], timeperiod=rsi_len)

    # OBV и его MA
    df["obv_val"] = talib.OBV(df["close"], df["volume"])
    df["obv_ma"] = talib.SMA(df["obv_val"], timeperiod=obv_ma_len)

    # ATR для фильтра волатильности
    df["atr_vol"] = talib.ATR(
        df["high"], df["low"], df["close"], timeperiod=atr_len_vol
    )
    df["avg_atr_vol"] = talib.SMA(df["atr_vol"], timeperiod=atr_len_vol * 2)

    # HTF EMA (Прокси: длинная EMA на текущем ТФ, смещенная на 1 бар)
    # Смещение на 1 бар имитирует использование закрытого бара или более высокого ТФ.
    df["htf_ema"] = talib.EMA(df["close"], timeperiod=htf_ema_len).shift(1)

    # ATR для управления выходом (SL/TP)
    df["atr_exit"] = talib.ATR(
        df["high"], df["low"], df["close"], timeperiod=atr_exit_len
    )

    # Заполняем NaN, которые появляются в начале из-за расчетов индикаторов
    df = df.dropna().reset_index(drop=True)
    return df


# Функция для расчёта ликвидационной цены (без изменений)
def calculate_liquidation_price(side, entry_price, leverage, maker_fee, taker_fee):
    """Рассчитывает приблизительную цену ликвидации."""
    mmr = 0.006  # Maintenance Margin Rate (0.6%)
    # Формула для лонга
    if side == "BUY":
        # Упрощенная формула, не учитывающая точно комиссии, но близкая к реальности
        return entry_price * (1 - (1 / leverage) + mmr) / (1 + taker_fee + maker_fee)
    # Формула для шорта
    else:  # SELL
        return entry_price * (1 + (1 / leverage) - mmr) / (1 - taker_fee - maker_fee)


# Этап 3: Симуляция сделок со всеми условиями стратегии
def simulate_trades_advanced(
    df,
    initial_capital,
    leverage,
    maker_fee,
    taker_fee,
    risk_per_trade_percent,
    rsi_ob_level,
    rsi_os_level,
    stop_loss_atr_mult,
    take_profit_atr_mult,
    use_volatility_filter,
    min_atr_multiplier,
    use_htf_filter,
    use_rsi_filter_entry,
    use_obv_filter_entry,
):
    """Симуляция торговли по заданной стратегии."""
    current_equity = initial_capital
    current_trade = None
    closed_trades = []

    # Начинаем с 1-го бара (индекс 1), чтобы безопасно использовать i-1 для кроссоверов
    for i in range(1, len(df)):
        current_time = df.iloc[i]["timestamp"]
        current_open = df.iloc[i]["open"]
        current_high = df.iloc[i]["high"]
        current_low = df.iloc[i]["low"]
        current_close = df.iloc[i]["close"]

        # Данные предыдущего бара (i-1)
        prev_ema_fast = df.iloc[i - 1]["ema_fast"]
        prev_ema_slow = df.iloc[i - 1]["ema_slow"]

        # Значения индикаторов для текущего бара (i)
        ema_fast = df.iloc[i]["ema_fast"]
        ema_slow = df.iloc[i]["ema_slow"]
        vwap_val = df.iloc[i]["vwap_val"]
        rsi_val = df.iloc[i]["rsi_val"]
        obv_val = df.iloc[i]["obv_val"]
        obv_ma = df.iloc[i]["obv_ma"]
        atr_vol = df.iloc[i]["atr_vol"]
        avg_atr_vol = df.iloc[i]["avg_atr_vol"]
        htf_ema = df.iloc[i]["htf_ema"]
        atr_exit = df.iloc[i]["atr_exit"]

        # --- Применение Фильтров ---
        # 1. Фильтр волатильности
        is_volatile_enough = (
            (atr_vol > (avg_atr_vol * min_atr_multiplier))
            if use_volatility_filter
            else True
        )

        # 2. Фильтр старшего таймфрейма (HTF)
        is_htf_bullish = (current_close > htf_ema) if use_htf_filter else True
        is_htf_bearish = (current_close < htf_ema) if use_htf_filter else True

        # 3. Дополнительные фильтры входа (RSI/OBV)
        rsi_long_filter = (rsi_val < rsi_ob_level) if use_rsi_filter_entry else True
        obv_long_filter = (obv_val > obv_ma) if use_obv_filter_entry else True

        rsi_short_filter = (rsi_val > rsi_os_level) if use_rsi_filter_entry else True
        obv_short_filter = (obv_val < obv_ma) if use_obv_filter_entry else True

        # --- Условия Сигналов (Long/Short Entry) ---
        # Базовое условие: Кроссовер/Кроссандер
        long_entry_cond_base = (ema_fast > ema_slow) and (
            prev_ema_fast <= prev_ema_slow
        )
        short_entry_cond_base = (ema_fast < ema_slow) and (
            prev_ema_fast >= prev_ema_slow
        )

        # Расширенные условия с опциональными фильтрами (Исправлен синтаксис!)
        long_entry_cond = (
            long_entry_cond_base
            and (current_close > vwap_val)
            and rsi_long_filter
            and obv_long_filter
            and is_htf_bullish
            and is_volatile_enough
        )

        short_entry_cond = (
            short_entry_cond_base
            and (current_close < vwap_val)
            and rsi_short_filter
            and obv_short_filter
            and is_htf_bearish
            and is_volatile_enough
        )

        # --- Логика Закрытия Сделки ---
        if current_trade:
            side = current_trade["side"]
            entry_price = current_trade["entry_price"]
            entry_atr_at_open = current_trade["entry_atr"]
            exit_price = None
            close_reason = None

            # Расчет TP/SL для текущей открытой сделки
            # Расстояние SL/TP основано на ATR, зафиксированном при входе
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
                exit_price = stop_loss_price  # Закрываем по SL цене
                close_reason = "Stop Loss"

            # 2. Проверка Тейк-Профита (если SL еще не сработал)
            if close_reason is None:
                if (side == "BUY" and current_high >= take_profit_price) or (
                    side == "SELL" and current_low <= take_profit_price
                ):
                    exit_price = take_profit_price  # Закрываем по TP цене
                    close_reason = "Take Profit"

            # 3. Проверка обратного сигнала (если SL/TP не сработали)
            if close_reason is None:
                if (side == "BUY" and short_entry_cond) or (
                    side == "SELL" and long_entry_cond
                ):
                    exit_price = current_close
                    close_reason = "Reverse Signal"

            # 4. Проверка ликвидации (крайний случай)
            lq_price = current_trade["liquidation_price"]
            if close_reason is None:
                if (side == "BUY" and current_low <= lq_price) or (
                    side == "SELL" and current_high >= lq_price
                ):
                    exit_price = lq_price
                    close_reason = "Liquidation"

            if close_reason is not None:
                # Рассчитываем PNL и обновляем капитал
                # NOTE: При скальпинге PNL = (exit - entry) * qty * leverage - fees

                # Комиссии за вход и выход
                fee_entry = (
                    current_trade["position_qty"]
                    * entry_price
                    * (maker_fee + taker_fee)
                )
                fee_exit = (
                    current_trade["position_qty"] * exit_price * (maker_fee + taker_fee)
                )

                if side == "BUY":
                    # PNL для лонга
                    pnl = (
                        (exit_price - entry_price)
                        * current_trade["position_qty"]
                        * leverage
                        - fee_entry
                        - fee_exit
                    )
                else:  # SELL
                    # PNL для шорта
                    pnl = (
                        (entry_price - exit_price)
                        * current_trade["position_qty"]
                        * leverage
                        - fee_entry
                        - fee_exit
                    )

                current_equity += pnl  # Обновляем капитал

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
                current_trade = None  # Закрываем сделку

        # --- Логика Открытия Сделки ---
        if not current_trade:  # Если нет открытой сделки

            # Динамический расчет размера позиции на основе риска
            position_qty = 0.0
            max_loss_usd = current_equity * (risk_per_trade_percent / 100)

            # Рассчитываем ATR для выхода (для определения стоп-лосса)
            # Расстояние SL в абсолютных долларах
            calculated_stop_loss_dist_usd = atr_exit * stop_loss_atr_mult

            if calculated_stop_loss_dist_usd > 0:
                # Количество, ограниченное риском
                risk_based_qty = max_loss_usd / calculated_stop_loss_dist_usd
                # Количество, ограниченное маржой (максимально возможное с плечом)
                margin_based_qty = (current_equity * leverage) / current_close
                # Берем наименьшее из двух, чтобы соблюсти управление риском
                position_qty = min(risk_based_qty, margin_based_qty)

            # Проверка, что сделка имеет смысл (комиссия не съест весь риск)
            estimated_total_commission = (
                position_qty * current_close * (maker_fee + taker_fee) * 2
            )  # Вход и выход
            # Комиссия не должна быть более 10% от потенциального убытка
            commission_check_ok = estimated_total_commission < (max_loss_usd * 0.1)

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
                        "entry_atr": atr_exit,  # Сохраняем ATR на момент входа
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": None,
                        "success": None,
                        "liquidation": False,
                        "close_reason": None,
                        "equity_at_entry": current_equity,
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
                        "entry_atr": atr_exit,  # Сохраняем ATR на момент входа
                        "exit_time": None,
                        "exit_price": None,
                        "pnl": None,
                        "success": None,
                        "liquidation": False,
                        "close_reason": None,
                        "equity_at_entry": current_equity,
                    }

    # Если есть открытая сделка в конце данных, закрываем её по последней цене
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

    return pd.DataFrame(closed_trades)


# Этап 4: Анализ результатов
def analyze_results(trades_df, initial_capital):
    """Рассчитывает и отображает ключевые метрики бэктеста."""
    if trades_df.empty:
        print("Нет закрытых сделок для анализа.")
        return

    total_pnl = trades_df["pnl"].sum()

    # Успешность считаем как долю сделок с положительным PNL
    success_rate = (trades_df["pnl"] > 0).mean() * 100
    liquidation_rate = trades_df["liquidation"].mean() * 100
    num_trades = len(trades_df)

    final_equity = initial_capital + total_pnl
    return_on_capital = (final_equity - initial_capital) / initial_capital * 100

    print(f"\n--- Результаты Бэктеста ---")
    print(f"Начальный капитал: ${initial_capital:,.2f}")
    print(f"Финальный капитал: ${final_equity:,.2f}")
    print(f"Общий PNL: ${total_pnl:,.2f}")
    print(f"Общее количество сделок: {num_trades}")
    print(f"Прибыльность (% от начального капитала): {return_on_capital:.2f}%")
    print(f"Процент прибыльных сделок: {success_rate:.1f}%")
    print(f"Процент ликвидаций: {liquidation_rate:.1f}%")

    # Кумулятивный PNL
    plt.figure(figsize=(14, 7))
    plt.plot(
        trades_df["exit_time"],
        trades_df["equity_after_trade"],
        label="Кумулятивный Капитал",
        linewidth=2,
        color="blue",
    )
    plt.axhline(
        initial_capital, color="gray", linestyle="--", label="Начальный Капитал"
    )
    plt.title("Кумулятивный Капитал Стратегии")
    plt.xlabel("Время")
    plt.ylabel("Капитал (USDT)")
    plt.legend()
    plt.grid()
    plt.show()

    # Распределение PNL по сделкам
    plt.figure(figsize=(10, 6))
    plt.hist(trades_df["pnl"], bins=50, color="skyblue", edgecolor="black")
    plt.title("Распределение PNL по Сделкам")
    plt.xlabel("PNL (USDT)")
    plt.ylabel("Количество Сделок")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


# Основной скрипт
if __name__ == "__main__":
    # --- Параметры Стратегии ---
    initial_capital = 1000000.0  # Начальный капитал для симуляции
    leverage = 10.0  # Плечо
    maker_fee = 0.0002  # Комиссия мейкера (0.02%)
    taker_fee = 0.0005  # Комиссия тейкера (0.05%)
    risk_per_trade_percent = 0.25  # Риск на сделку в % от капитала

    # Параметры индикаторов
    ema_fast_len = 9
    ema_slow_len = 21
    rsi_len = 14
    rsi_ob_level = 70
    rsi_os_level = 30
    obv_ma_len = 10

    # Фильтры
    use_volatility_filter = True
    atr_len_vol = 20
    min_atr_multiplier = 1.2

    use_htf_filter = True
    htf_ema_len = 50

    # Опциональные фильтры входа
    use_rsi_filter_entry = False
    use_obv_filter_entry = False

    # Параметры выхода
    atr_exit_len = 14
    stop_loss_atr_mult = 1.5
    take_profit_atr_mult = 2.0

    # --- Загрузка и расчет данных ---
    df = load_and_analyze_data(file_path)
    df = calculate_all_strategy_components(
        df.copy(),  # Передаем копию, чтобы не изменять исходный df
        ema_fast_len,
        ema_slow_len,
        rsi_len,
        obv_ma_len,
        atr_len_vol,
        htf_ema_len,
        atr_exit_len,
    )

    # --- Симуляция сделок ---
    trades_df = simulate_trades_advanced(
        df,
        initial_capital,
        leverage,
        maker_fee,
        taker_fee,
        risk_per_trade_percent,
        rsi_ob_level,
        rsi_os_level,
        stop_loss_atr_mult,
        take_profit_atr_mult,
        use_volatility_filter,
        min_atr_multiplier,
        use_htf_filter,
        use_rsi_filter_entry,
        use_obv_filter_entry,
    )

    # --- Анализ результатов ---
    analyze_results(trades_df, initial_capital)
