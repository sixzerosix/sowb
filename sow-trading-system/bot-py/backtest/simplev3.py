import os
import pandas as pd
import talib
import numpy as np
from math import copysign

# Определяем путь к файлу (предполагая то же имя файла, что и в предыдущем запросе)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    script_dir,
    "BTC_USDT_1m_2025-07-01_00_00_00_to_2025-09-01_00_00_00.csv",
)

if not os.path.exists(file_path):
    print(f"Ошибка: Файл {file_path} не найден!")
    print("Убедитесь, что файл находится в той же папке, что и скрипт.")
    exit()


# === ЭТАП 1: Загрузка и анализ данных (без изменений) ===
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


# === ЭТАП 2: Расчет Индикаторов по Pine Script V3 ===
def calculate_pine_v3_indicators(df, ema_fast_len, ema_slow_len, rsi_fast_len, atr_len):
    """Расчет всех индикаторов, необходимых для логики Pine Script V3."""

    # 1. EMAs (9 и 21)
    df["ema_fast"] = talib.EMA(df["close"], timeperiod=ema_fast_len)
    df["ema_slow"] = talib.EMA(df["close"], timeperiod=ema_slow_len)

    # 2. VWAP Proxy (используем EMA 50, так как стандартный VWAP требует daily reset)
    df["vwap_val"] = talib.EMA(df["close"], timeperiod=50)

    # 3. RSI Fast (RSI 5)
    df["rsi_fast"] = talib.RSI(df["close"], timeperiod=rsi_fast_len)

    # 4. OBV и Delta (изменение OBV)
    # talib.OBV: Volume * sign(Close - Previous Close)
    df["obv_val"] = talib.OBV(df["close"], df["volume"])
    df["delta_obv"] = df["obv_val"].diff()

    # 5. ATR для стопов
    df["atr_exit"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=atr_len)

    # Удаляем строки с NaN, возникшие из-за расчетов
    df = df.dropna().reset_index(drop=True)
    return df


# Вспомогательная функция для расчета цены ликвидации (без изменений)
def calculate_liquidation_price(side, entry_price, leverage, maker_fee, taker_fee):
    """Рассчитывает приблизительную цену ликвидации."""
    mmr = 0.006
    if side == "BUY":
        return entry_price * (1 - (1 / leverage) + mmr) / (1 + taker_fee + maker_fee)
    else:  # SELL
        return entry_price * (1 + (1 / leverage) - mmr) / (1 - taker_fee - maker_fee)


# === ЭТАП 3: Симуляция Сделок по Enhanced Confluence ===
def simulate_trades_pine_v3(
    df,
    initial_capital,
    leverage,
    maker_fee,
    taker_fee,
    risk_per_trade_percent,
    rsi_mid_level,
    stop_loss_atr_mult,
    take_profit_atr_mult,  # Оставим TP, даже если в Pine не указан (для реалистичности)
):
    """Симуляция торговли по стратегии Pine Script V3 Enhanced."""
    current_equity = initial_capital
    current_trade = None
    closed_trades = []

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
        rsi_fast = df.iloc[i]["rsi_fast"]
        delta_obv = df.iloc[i]["delta_obv"]
        atr_exit = df.iloc[i]["atr_exit"]

        # === Условия Входа (Enhanced Confluence) ===
        ema_cross_up = (ema_fast > ema_slow) and (prev_ema_fast <= prev_ema_slow)
        ema_cross_down = (ema_fast < ema_slow) and (prev_ema_fast >= prev_ema_slow)

        # Long Entry: EMA Cross Up AND close > VWAP AND RSI5 > 50 AND Delta > 0
        long_entry_cond = (
            ema_cross_up
            and (current_close > vwap_val)
            and (rsi_fast > rsi_mid_level)
            and (delta_obv > 0)
        )

        # Short Entry: EMA Cross Down AND close < VWAP AND RSI5 < 50 AND Delta < 0
        short_entry_cond = (
            ema_cross_down
            and (current_close < vwap_val)
            and (rsi_fast < rsi_mid_level)
            and (delta_obv < 0)
        )

        # === Логика Закрытия Сделки ===
        if current_trade:
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
                # Проверка: если цена закрытия ниже SL, то закрываем по цене SL
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
    print(" " * 10 + "РЕЗУЛЬТАТЫ PINE SCRIPT V3 BACKTEST (Enhanced)")
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


# === ОСНОВНОЙ СКРИПТ ===
if __name__ == "__main__":

    # --- Параметры Финансирования и Риска ---
    initial_capital = 1000000.0
    leverage = 10.0
    maker_fee = 0.0002
    taker_fee = 0.0005
    risk_per_trade_percent = 0.25  # Используется для расчета размера позиции

    # --- Параметры Индикаторов (из Pine Script V3) ---
    ema_fast_len = 9
    ema_slow_len = 21
    rsi_fast_len = 5
    rsi_mid_level = 50  # RSI5 > 50 для Long, RSI5 < 50 для Short

    # VWAP Proxy: Используем EMA 50 как длинную, стабильную среднюю
    vwap_len = 50

    # --- Параметры Выхода (ATR Stops) ---
    atr_exit_len = 14
    stop_loss_atr_mult = 1.5  # 1.5 * ATR(14) для SL (из Pine Script)
    take_profit_atr_mult = (
        3.0  # Добавляем TP для реалистичности (можно поставить 0.0 для выключения)
    )

    # --- Загрузка и расчет данных ---
    df = load_and_analyze_data(file_path)
    df = calculate_pine_v3_indicators(
        df.copy(), ema_fast_len, ema_slow_len, rsi_fast_len, atr_exit_len
    )

    # --- Симуляция сделок ---
    trades_df = simulate_trades_pine_v3(
        df,
        initial_capital,
        leverage,
        maker_fee,
        taker_fee,
        risk_per_trade_percent,
        rsi_mid_level,
        stop_loss_atr_mult,
        take_profit_atr_mult,
    )

    # --- Анализ результатов ---
    analyze_results(trades_df, initial_capital)
