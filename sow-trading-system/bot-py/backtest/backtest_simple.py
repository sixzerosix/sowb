import os
import pandas as pd
import talib
import matplotlib.pyplot as plt

# Определяем путь к файлу
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "BTC_USDT_1m_2024-11-01 00_00_00_to_current.csv")

# Проверяем, существует ли файл
if not os.path.exists(file_path):
    print(f"Ошибка: Файл {file_path} не найден!")
    print("Убедитесь, что файл находится в той же папке, что и скрипт.")
    exit()


# Этап 1: Загрузка и анализ данных
def load_and_analyze_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Первые 5 строк:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())
    print("\nОписательная статистика:")
    print(df.describe())

    plt.figure(figsize=(14, 7))
    plt.plot(df["timestamp"], df["close"], label="Цена закрытия", linewidth=1)
    plt.title("График цены BTC/USDT (1 минута)")
    plt.xlabel("Время")
    plt.ylabel("Цена (USDT)")
    plt.legend()
    plt.grid()
    plt.show()

    return df


# Этап 2: Базовая стратегия (EMA 9 и EMA 21)
def calculate_signals(df):
    df["ema9"] = talib.EMA(df["close"], timeperiod=9)
    df["ema21"] = talib.EMA(df["close"], timeperiod=21)
    df["signal"] = 0
    df.loc[df["ema9"] > df["ema21"], "signal"] = 1  # Сигнал на покупку
    df.loc[df["ema9"] < df["ema21"], "signal"] = -1  # Сигнал на продажу

    plt.figure(figsize=(14, 7))
    plt.plot(df["timestamp"], df["close"], label="Цена закрытия", linewidth=1)
    plt.plot(df["timestamp"], df["ema9"], label="EMA 9", linestyle="--", linewidth=1)
    plt.plot(df["timestamp"], df["ema21"], label="EMA 21", linestyle="--", linewidth=1)
    plt.scatter(
        df[df["signal"] == 1]["timestamp"],
        df[df["signal"] == 1]["close"],
        label="Покупка",
        marker="^",
        color="g",
        alpha=1,
    )
    plt.scatter(
        df[df["signal"] == -1]["timestamp"],
        df[df["signal"] == -1]["close"],
        label="Продажа",
        marker="v",
        color="r",
        alpha=1,
    )
    plt.title("Сигналы стратегии")
    plt.xlabel("Время")
    plt.ylabel("Цена (USDT)")
    plt.legend()
    plt.grid()
    plt.show()

    return df


# Функция для расчёта ликвидационной цены (без изменений)
def calculate_liquidation_price(side, entry_price, leverage, maker_fee, taker_fee):
    mmr = 0.006  # Maintenance Margin Rate (0.6%)
    if side == "BUY":
        return entry_price * (1 - (1 / leverage) + mmr) / (1 + taker_fee + maker_fee)
    else:
        return entry_price * (1 + (1 / leverage) - mmr) / (1 - taker_fee - maker_fee)


# Этап 3: Симуляция сделок с TP/SL и условием входа только после закрытия предыдущей сделки
def simulate_trades(
    df, leverage, maker_fee, taker_fee, position_size, take_profit, stop_loss
):
    current_trade = None
    closed_trades = []

    for i in range(len(df)):
        current_time = df.iloc[i]["timestamp"]
        current_price = df.iloc[i]["close"]
        current_low = df.iloc[i]["low"]
        current_high = df.iloc[i]["high"]
        signal = df.iloc[i]["signal"]

        # Если сделка уже открыта, проверяем условия для её закрытия
        if current_trade:
            side = current_trade["side"]
            lq_price = current_trade["liquidation_price"]
            entry_price = current_trade["entry_price"]
            exit_price = None
            liquidation_hit = False
            close_trade = False

            # 1. Ликвидация
            if (side == "BUY" and current_low <= lq_price) or (
                side == "SELL" and current_high >= lq_price
            ):
                exit_price = lq_price
                liquidation_hit = True
                close_trade = True

            # 2. Проверка TP/SL (для длинных и коротких позиций)
            if not close_trade:
                if side == "BUY":
                    if current_price >= entry_price * (1 + take_profit):
                        exit_price = current_price
                        close_trade = True
                    elif current_price <= entry_price * (1 - stop_loss):
                        exit_price = current_price
                        close_trade = True
                else:  # для коротких позиций
                    if current_price <= entry_price * (1 - take_profit):
                        exit_price = current_price
                        close_trade = True
                    elif current_price >= entry_price * (1 + stop_loss):
                        exit_price = current_price
                        close_trade = True

            # 3. Проверка смены сигнала (если ни ликвидация, ни TP/SL не сработали)
            if not close_trade:
                if (side == "BUY" and signal == -1) or (side == "SELL" and signal == 1):
                    exit_price = current_price
                    close_trade = True

            if close_trade:
                fee = current_trade["fee_paid"]
                if side == "BUY":
                    pnl = (exit_price - entry_price) * position_size * leverage - fee
                else:
                    pnl = (entry_price - exit_price) * position_size * leverage - fee

                # Сделку считаем успешной, если PNL положительный (а не просто отсутствие ликвидации)
                success = pnl > 0

                current_trade.update(
                    {
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "success": success,
                        "liquidation": liquidation_hit,
                    }
                )
                closed_trades.append(current_trade)
                current_trade = None  # Закрываем сделку

        # Если нет открытой сделки, и сигнал не равен 0, открываем новую сделку
        # (условие гарантирует, что новая сделка не войдет до закрытия предыдущей)
        if not current_trade and signal != 0:
            side = "BUY" if signal == 1 else "SELL"
            entry_price = current_price
            lq_price = calculate_liquidation_price(
                side, entry_price, leverage, maker_fee, taker_fee
            )
            fee = position_size * (maker_fee + taker_fee) * leverage

            current_trade = {
                "entry_time": current_time,
                "entry_price": entry_price,
                "side": side,
                "liquidation_price": lq_price,
                "fee_paid": fee,
                "exit_time": None,
                "exit_price": None,
                "pnl": None,
                "success": None,
                "liquidation": False,
            }

    return pd.DataFrame(closed_trades)


# Этап 4: Анализ результатов
def analyze_results(trades_df):
    total_pnl = trades_df["pnl"].sum()
    total_fees = trades_df["fee_paid"].sum()
    # Успешность считаем как долю сделок с положительным PNL
    if len(trades_df) > 0:
        success_rate = (trades_df["pnl"] > 0).mean() * 100
        liquidation_rate = trades_df["liquidation"].mean() * 100
    else:
        success_rate = liquidation_rate = 0

    print(f"Общий PNL: ${total_pnl:.2f}")
    print(f"Общие комиссии: ${total_fees:.2f}")
    print(f"Процент прибыльных сделок: {success_rate:.1f}%")
    print(f"Процент ликвидаций: {liquidation_rate:.1f}%")

    plt.figure(figsize=(14, 7))
    plt.plot(
        trades_df["exit_time"],
        trades_df["pnl"].cumsum(),
        label="Кумулятивный PNL",
        linewidth=1,
    )
    plt.title("Кумулятивный PNL")
    plt.xlabel("Время")
    plt.ylabel("PNL (USDT)")
    plt.legend()
    plt.grid()
    plt.show()


# Основной скрипт
if __name__ == "__main__":
    leverage = 10  # Плечо
    maker_fee = 0.0002  # Комиссия мейкера (0.02%)
    taker_fee = 0.0005  # Комиссия тейкера (0.05%)
    position_size = 1.0  # Размер позиции в USDT

    # Уровни TP и SL (относительно цены входа)
    take_profit = 0.05  # TP: +50%
    stop_loss = 0.03  # SL: -30%

    df = load_and_analyze_data(file_path)
    df = calculate_signals(df)
    trades_df = simulate_trades(
        df, leverage, maker_fee, taker_fee, position_size, take_profit, stop_loss
    )
    analyze_results(trades_df)
