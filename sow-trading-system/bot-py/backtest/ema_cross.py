import os
import pandas as pd
import talib
import json
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Ошибка: Файл config.json не найден в директории {script_dir}")
    exit()

global_config = config["global"]
coins_config = config["coins"]

symbol = "BTC/USDT"
try:
    coin_config = next(c for c in coins_config if c["symbol"] == symbol)
except StopIteration:
    print(f"Ошибка: Символ {symbol} не найден в конфигурации")
    exit()

# Параметры стратегии
tp_pct = coin_config.get("tp_pct", global_config["tp_pct"])
sl_pct = coin_config.get("sl_pct", global_config["sl_pct"])
leverage = min(coin_config.get("leverage", global_config["leverage"]), 100)
maker_fee = coin_config.get("maker_commission", global_config["maker_commission"]) / 100
taker_fee = coin_config.get("taker_commission", global_config["taker_commission"]) / 100
position_size = 1.0

csv_path = os.path.join(script_dir, "./BTC_USDT_1m_2024-01-01 00_00_00_to_current.csv")
if not os.path.exists(csv_path):
    print(f"Ошибка: Файл данных {csv_path} не найден!")
    exit()

df = pd.read_csv(csv_path, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Расчет индикаторов
df["ema9"] = talib.EMA(df["close"], timeperiod=9)
df["ema21"] = talib.EMA(df["close"], timeperiod=21)
df["prev_ema9"] = df["ema9"].shift(1)
df["prev_ema21"] = df["ema21"].shift(1)
cross_up = (df["ema9"] > df["ema21"]) & (df["prev_ema9"] <= df["prev_ema21"])
cross_down = (df["ema9"] < df["ema21"]) & (df["prev_ema9"] >= df["prev_ema21"])


def calculate_liquidation_price(side, entry_price, leverage, maker_fee, taker_fee):
    mmr = 0.006  # Maintenance Margin Rate
    if side == "BUY":
        return entry_price * (1 - (1 / leverage) + mmr) / (1 + taker_fee + maker_fee)
    else:
        return entry_price * (1 + (1 / leverage) - mmr) / (1 - taker_fee - maker_fee)


current_trade = None
closed_trades = []
equity_curve = []
max_drawdown = 0
peak = 0

for i in range(len(df)):
    current_time = df.iloc[i]["timestamp"]
    current_price = df.iloc[i]["close"]
    current_low = df.iloc[i]["low"]
    current_high = df.iloc[i]["high"]

    # Закрытие сделки
    if current_trade:
        side = current_trade["side"]
        lq_price = current_trade["liquidation_price"]
        exit_price = None
        liquidation_hit = False

        # Проверка ликвидации
        if (side == "BUY" and current_low <= lq_price) or (
            side == "SELL" and current_high >= lq_price
        ):
            exit_price = lq_price
            liquidation_hit = True
            close_trade = True
        else:
            # Проверка TP/SL
            if side == "BUY":
                tp = current_trade["tp_price"]
                sl = current_trade["sl_price"]
                close_trade = (tp_pct and current_high >= tp) or (
                    sl_pct and current_low <= sl
                )
                exit_price = (
                    tp
                    if (tp and current_high >= tp)
                    else (sl if (sl and current_low <= sl) else None)
                )
            else:
                tp = current_trade["tp_price"]
                sl = current_trade["sl_price"]
                close_trade = (tp_pct and current_low <= tp) or (
                    sl_pct and current_high >= sl
                )
                exit_price = (
                    tp
                    if (tp and current_low <= tp)
                    else (sl if (sl and current_high >= sl) else None)
                )

        if close_trade and exit_price:
            entry_price = current_trade["entry_price"]
            fee = current_trade["fee_paid"]

            if side == "BUY":
                pnl = (exit_price - entry_price) * position_size * leverage - fee
            else:
                pnl = (entry_price - exit_price) * position_size * leverage - fee

            current_trade.update(
                {
                    "exit_time": current_time,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "success": not liquidation_hit,
                    "liquidation": liquidation_hit,
                }
            )

            closed_trades.append(current_trade)
            current_trade = None

    # Открытие сделки
    if not current_trade:
        if cross_up.iloc[i]:
            side = "BUY"
            entry_price = current_price
            tp_price = (
                entry_price * (1 + (tp_pct / (100 * leverage))) if tp_pct else None
            )
            sl_price = (
                entry_price * (1 - (sl_pct / (100 * leverage))) if sl_pct else None
            )
            lq_price = calculate_liquidation_price(
                "BUY", entry_price, leverage, maker_fee, taker_fee
            )
        elif cross_down.iloc[i]:
            side = "SELL"
            entry_price = current_price
            tp_price = (
                entry_price * (1 - (tp_pct / (100 * leverage))) if tp_pct else None
            )
            sl_price = (
                entry_price * (1 + (sl_pct / (100 * leverage))) if sl_pct else None
            )
            lq_price = calculate_liquidation_price(
                "SELL", entry_price, leverage, maker_fee, taker_fee
            )
        else:
            continue

        current_trade = {
            "entry_time": current_time,
            "entry_price": entry_price,
            "side": side,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "liquidation_price": lq_price,
            "fee_paid": position_size * (maker_fee + taker_fee) * leverage,
            "exit_time": None,
            "exit_price": None,
            "pnl": None,
            "success": None,
            "liquidation": False,
        }

# Расчет статистики
trades_df = pd.DataFrame(closed_trades)
if not trades_df.empty:
    total_pnl = trades_df["pnl"].sum()
    total_fees = trades_df["fee_paid"].sum()
    successful = trades_df["success"].sum()
    failed = len(trades_df) - successful - trades_df["liquidation"].sum()
    liquidated = trades_df["liquidation"].sum()

    # Расчет просадки
    equity = trades_df["pnl"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = drawdown.min()

    # Волатильность
    returns = trades_df["pnl"] / position_size
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
else:
    total_pnl = 0
    total_fees = 0
    successful = failed = liquidated = 0
    max_drawdown = 0
    sharpe_ratio = 0

# Временные параметры
start_time = df["timestamp"].min()
end_time = df["timestamp"].max()
total_hours = (end_time - start_time).total_seconds() / 3600
actual_days = total_hours / 24

results = {
    "Размер позиции": f"${position_size:.2f}",
    "Плечо": f"{leverage}x",
    "Общий PNL": f"${total_pnl:.2f}",
    "Общие комиссии": f"${total_fees:.2f}",
    "Средний PNL на сделку": (
        f"${total_pnl/len(trades_df):.2f}" if len(trades_df) > 0 else "$0.00"
    ),
    "Успешные сделки": (
        f"{successful} ({successful/len(trades_df)*100:.1f}%)"
        if len(trades_df) > 0
        else "0"
    ),
    "Стоп-лосс сделки": f"{failed}",
    "Ликвидации": f"{liquidated}",
    "Максимальная просадка": f"${max_drawdown:.2f}",
    "Коэф. Шарпа": f"{sharpe_ratio:.2f}",
    "Фактический период": f"{actual_days:.1f} дней",
    "Доход/день": f"${total_pnl/actual_days:.2f}" if actual_days > 0 else "$0.00",
}

results_df = pd.DataFrame([results])
print(results_df)
results_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")
