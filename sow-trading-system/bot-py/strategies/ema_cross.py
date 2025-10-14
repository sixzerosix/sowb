import asyncio
from datetime import datetime
from indicators.indicators import calculate_indicators, check_signals
from telegram.bot import send_telegram_message
from database.db import log_trade, close_trade
from exchange.exchange import fetch_data
from utils.helpers import format_entry_message, format_exit_message


async def monitor_coin(coin_config, global_config, db, exchange_client):
    symbol = coin_config.get("symbol")
    timeframe = coin_config.get("timeframe", global_config.get("timeframe"))
    limit = coin_config.get("limit", global_config.get("limit"))
    leverage = coin_config.get("leverage", global_config.get("leverage"))
    maker_commission = coin_config.get(
        "maker_commission", global_config.get("maker_commission")
    )
    taker_commission = coin_config.get(
        "taker_commission", global_config.get("taker_commission")
    )
    tp_pct = coin_config.get("tp_pct", global_config.get("tp_pct"))
    sl_pct = coin_config.get("sl_pct", global_config.get("sl_pct"))
    update_frequency = global_config.get("update_frequency", 60)
    telegram_config = global_config.get("telegram", {})

    open_trade = None
    trade_id = None

    while True:
        df = await fetch_data(exchange_client, symbol, timeframe, limit)
        if df is None or df.empty:
            print(
                f"{datetime.utcnow().isoformat()} - Не удалось получить данные для {symbol}. Повтор через 10 секунд."
            )
            await asyncio.sleep(10)
            continue

        df = calculate_indicators(df)
        cross_up, cross_down = check_signals(df)
        latest_price = df["close"].iloc[-1]
        print(f"Symbol: {symbol} | Price: {latest_price:.2f} | Timeframe: {timeframe}")

        if open_trade is None:
            if cross_up or cross_down:
                side = "BUY" if cross_up else "SELL"
                signal_type = "EMA Cross UP" if cross_up else "EMA Cross DOWN"
                entry_time = datetime.utcnow().isoformat()

                if side == "BUY":
                    tp_price = latest_price * (1 + (tp_pct / (100 * leverage)))
                    sl_price = latest_price * (1 - (sl_pct / (100 * leverage)))
                else:
                    tp_price = latest_price * (1 - (tp_pct / (100 * leverage)))
                    sl_price = latest_price * (1 + (sl_pct / (100 * leverage)))

                trade = {
                    "timestamp": entry_time,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "side": side,
                    "entry_price": latest_price,
                    "exit_price": None,
                    "quantity": 1.0,
                    "pnl": None,
                    "status": "OPEN",
                    "signal_type": signal_type,
                    "leverage": leverage,
                    "maker_commission": maker_commission,
                    "taker_commission": taker_commission,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                }
                trade_id = await log_trade(db, trade)
                open_trade = trade

                entry_msg = format_entry_message(trade)
                await send_telegram_message(
                    telegram_config.get("bot_token"),
                    telegram_config.get("chat_id"),
                    entry_msg,
                )
                print(
                    f"{datetime.utcnow().isoformat()} - Отправлено сообщение в Telegram (вход):\n{entry_msg}"
                )
        else:
            if open_trade["side"] == "BUY":
                if (
                    latest_price >= open_trade["tp_price"]
                    or latest_price <= open_trade["sl_price"]
                ):
                    exit_price = latest_price
                    pnl = (exit_price - open_trade["entry_price"]) * open_trade[
                        "quantity"
                    ]
                    await close_trade(db, trade_id, exit_price, pnl)
                    exit_msg = format_exit_message(open_trade, exit_price, pnl)
                    await send_telegram_message(
                        telegram_config.get("bot_token"),
                        telegram_config.get("chat_id"),
                        exit_msg,
                    )
                    print(
                        f"{datetime.utcnow().isoformat()} - Отправлено сообщение в Telegram (выход):\n{exit_msg}"
                    )
                    open_trade = None
                    trade_id = None
            elif open_trade["side"] == "SELL":
                if (
                    latest_price <= open_trade["tp_price"]
                    or latest_price >= open_trade["sl_price"]
                ):
                    exit_price = latest_price
                    pnl = (open_trade["entry_price"] - exit_price) * open_trade[
                        "quantity"
                    ]
                    await close_trade(db, trade_id, exit_price, pnl)
                    exit_msg = format_exit_message(open_trade, exit_price, pnl)
                    await send_telegram_message(
                        telegram_config.get("bot_token"),
                        telegram_config.get("chat_id"),
                        exit_msg,
                    )
                    print(
                        f"{datetime.utcnow().isoformat()} - Отправлено сообщение в Telegram (выход):\n{exit_msg}"
                    )
                    open_trade = None
                    trade_id = None

        await asyncio.sleep(update_frequency)
