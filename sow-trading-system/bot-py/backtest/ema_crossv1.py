import os
import pandas as pd
import talib
import json
from datetime import datetime

# Получаем путь к текущей директории скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Загрузка конфигурации с полным путем
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Ошибка: Файл config.json не найден в директории {script_dir}")
    print("Убедитесь, что файл находится в той же папке, что и скрипт!")
    exit()

global_config = config["global"]
coins_config = config["coins"]

# Выбор символа для тестирования
symbol = "BTC/USDT"
try:
    coin_config = next(c for c in coins_config if c["symbol"] == symbol)
except StopIteration:
    print(f"Ошибка: Символ {symbol} не найден в конфигурации")
    exit()

# Получаем параметры с приоритетом настроек монеты
tp_pct = coin_config.get("tp_pct", global_config["tp_pct"])
sl_pct = coin_config.get("sl_pct", global_config["sl_pct"])
leverage = coin_config.get("leverage", global_config["leverage"])
maker_fee = coin_config.get("maker_commission", global_config["maker_commission"]) / 100
taker_fee = coin_config.get("taker_commission", global_config["taker_commission"]) / 100

position_size = 1.0  # Размер позиции в USDT

# Проверка существования CSV файла
csv_path = os.path.join(script_dir, "BTC_USDT_1m_2024-11-01 00_00_00_to_current.csv")
if not os.path.exists(csv_path):
    print(f"Ошибка: Файл данных {csv_path} не найден!")
    exit()

# Загрузка данных из CSV
df = pd.read_csv(csv_path, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Расчет индикаторов
df["ema9"] = talib.EMA(df["close"], timeperiod=9)
df["ema21"] = talib.EMA(df["close"], timeperiod=21)

# Поиск пересечений EMA
df["prev_ema9"] = df["ema9"].shift(1)
df["prev_ema21"] = df["ema21"].shift(1)
cross_up = (df["ema9"] > df["ema21"]) & (df["prev_ema9"] <= df["prev_ema21"])
cross_down = (df["ema9"] < df["ema21"]) & (df["prev_ema9"] >= df["prev_ema21"])


# Функция расчета ликвидационной цены
def calculate_liquidation_price(side, entry_price, leverage, maker_fee, taker_fee):
    """
    Расчет ликвидационной цены для изолированной маржи
    Формула: https://www.bybit.com/help-center/article/How-to-Calculate-Liquidation-Price
    """
    mmr = 0.006  # Maintenance Margin Rate (0.6% для BTC/USDT)

    if side == "BUY":
        liquidation_price = (
            entry_price * (1 - (1 / leverage) + mmr) / (1 + taker_fee + maker_fee)
        )
    else:
        liquidation_price = (
            entry_price * (1 + (1 / leverage) - mmr) / (1 - taker_fee - maker_fee)
        )

    return liquidation_price


# Симуляция сделок
current_trade = None
closed_trades = []

for i in range(len(df)):
    current_time = df.iloc[i]["timestamp"]
    current_price = df.iloc[i]["close"]
    current_low = df.iloc[i]["low"]
    current_high = df.iloc[i]["high"]

    # Закрытие текущей сделки
    if current_trade is not None:
        side = current_trade["side"]
        liquidation_price = current_trade["liquidation_price"]
        liquidation_hit = False
        close_trade = False
        exit_price = None

        # Проверка ликвидации
        if side == "BUY" and current_low <= liquidation_price:
            close_trade = True
            exit_price = liquidation_price
            liquidation_hit = True
        elif side == "SELL" and current_high >= liquidation_price:
            close_trade = True
            exit_price = liquidation_price
            liquidation_hit = True
        else:
            # Проверка TP/SL
            if side == "BUY":
                if tp_pct and current_price >= current_trade["tp_price"]:
                    close_trade = True
                    exit_price = current_price
                if sl_pct and current_price <= current_trade["sl_price"]:
                    close_trade = True
                    exit_price = current_price
            else:
                if tp_pct and current_price <= current_trade["tp_price"]:
                    close_trade = True
                    exit_price = current_price
                if sl_pct and current_price >= current_trade["sl_price"]:
                    close_trade = True
                    exit_price = current_price

        if close_trade:
            # Рассчет PNL с учетом комиссий и плеча
            entry_price = current_trade["entry_price"]
            if side == "BUY":
                pnl = (exit_price - entry_price) * position_size * leverage
            else:
                pnl = (entry_price - exit_price) * position_size * leverage

            # Вычитаем комиссии
            pnl -= current_trade["fee_paid"]

            current_trade.update(
                {
                    "exit_time": current_time,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "success": not liquidation_hit
                    and (
                        (
                            exit_price >= current_trade["tp_price"]
                            if side == "BUY"
                            else exit_price <= current_trade["tp_price"]
                        )
                        if tp_pct
                        else False
                    ),
                    "liquidation": liquidation_hit,
                }
            )
            closed_trades.append(current_trade)
            current_trade = None

    # Открытие новой сделки
    if current_trade is None:
        if cross_up.iloc[i]:
            entry_price = current_price
            tp_price = (
                entry_price * (1 + (tp_pct / (100 * leverage))) if tp_pct else None
            )
            sl_price = (
                entry_price * (1 - (sl_pct / (100 * leverage))) if sl_pct else None
            )
            liquidation_price = calculate_liquidation_price(
                "BUY", entry_price, leverage, maker_fee, taker_fee
            )
            current_trade = {
                "entry_time": current_time,
                "entry_price": entry_price,
                "exit_time": None,
                "exit_price": None,
                "side": "BUY",
                "tp_price": tp_price,
                "sl_price": sl_price,
                "liquidation_price": liquidation_price,
                "pnl": None,
                "success": None,
                "liquidation": False,
                "fee_paid": position_size * (maker_fee + taker_fee),
            }
        elif cross_down.iloc[i]:
            entry_price = current_price
            tp_price = (
                entry_price * (1 - (tp_pct / (100 * leverage))) if tp_pct else None
            )
            sl_price = (
                entry_price * (1 + (sl_pct / (100 * leverage))) if sl_pct else None
            )
            liquidation_price = calculate_liquidation_price(
                "SELL", entry_price, leverage, maker_fee, taker_fee
            )
            current_trade = {
                "entry_time": current_time,
                "entry_price": entry_price,
                "exit_time": None,
                "exit_price": None,
                "side": "SELL",
                "tp_price": tp_price,
                "sl_price": sl_price,
                "liquidation_price": liquidation_price,
                "pnl": None,
                "success": None,
                "liquidation": False,
                "fee_paid": position_size * (maker_fee + taker_fee),
            }

# Закрытие последней сделки (если осталась открытой)
if current_trade is not None:
    current_trade["exit_time"] = df.iloc[-1]["timestamp"]
    current_trade["exit_price"] = df.iloc[-1]["close"]
    pnl = (
        (current_trade["exit_price"] - current_trade["entry_price"])
        * position_size
        * leverage
        if current_trade["side"] == "BUY"
        else (current_trade["entry_price"] - current_trade["exit_price"])
        * position_size
        * leverage
    )
    pnl -= current_trade["fee_paid"]
    current_trade["pnl"] = pnl
    current_trade["success"] = False  # Считаем просроченные сделки неуспешными
    closed_trades.append(current_trade)

# Расчет статистики
trades_df = pd.DataFrame(closed_trades)
total_trades = len(trades_df)
successful = trades_df["success"].sum()
failed = total_trades - successful
liquidated = trades_df["liquidation"].sum()

if total_trades > 0:
    avg_duration = (
        trades_df["exit_time"] - trades_df["entry_time"]
    ).dt.total_seconds().mean() / 60
else:
    avg_duration = 0

total_pnl = trades_df["pnl"].sum()

# Расчет времени теста
start_time = df["timestamp"].min()
end_time = df["timestamp"].max()
total_hours = (end_time - start_time).total_seconds() / 3600

# Расчет доходности
periods = {
    "hour": 1,
    "12h": 12,
    "24h": 24,
    "week": 24 * 7,
    "month": 24 * 30,
    "year": 24 * 365,
}

income = {}
if total_hours > 0:
    hourly = total_pnl / total_hours
    for period, hours in periods.items():
        income[period] = hourly * hours
else:
    for period in periods:
        income[period] = 0

# Формирование таблицы
results = {
    "Размер позиции": f"${position_size:.2f}",
    "Среднее время сделки (мин)": f"{avg_duration:.1f}",
    "💸 Тейк (%)": tp_pct,
    "🛑 Стоп (%)": sl_pct,
    "✔️ Успешные сделки": successful,
    "❌ Не успешные сделки": failed,
    "💀 Ликвидации": liquidated,
    "Доход за час": f"${income['hour']:.2f}",
    "Доход за 12 часов": f"${income['12h']:.2f}",
    "Доход за 24 часа": f"${income['24h']:.2f}",
    "Доход за 1 неделю": f"${income['week']:.2f}",
    "Доход за 1 месяц": f"${income['month']:.2f}",
    "Доход в 1 год": f"${income['year']:.2f}",
}

results_df = pd.DataFrame([results])
print(results_df)
results_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")
