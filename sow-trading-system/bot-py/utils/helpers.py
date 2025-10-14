from datetime import datetime
import asyncio


def format_entry_message(trade):
    msg = (
        "🚀 *Trade Entry Signal* 🚀\n"
        f"*Symbol:* {trade['symbol']}\n"
        f"*Timeframe:* {trade['timeframe']}\n"
        f"*Signal:* {trade['signal_type']}\n"
        f"*Side:* {trade['side']}\n"
        f"*Entry Price:* {trade['entry_price']:.2f}\n"
        f"*TP (%):* {trade['tp_pct']}% -> Price: {trade['tp_price']:.2f}\n"
        f"*SL (%):* {trade['sl_pct']}% -> Price: {trade['sl_price']:.2f}\n"
        f"*Leverage:* {trade['leverage']}x\n"
        f"*Maker Fee:* {trade['maker_commission']}%\n"
        f"*Taker Fee:* {trade['taker_commission']}%\n"
        f"*Time:* {trade['timestamp']}\n"
    )
    return msg


def format_exit_message(trade, exit_price, pnl):
    emoji = "✅" if pnl > 0 else "❌"
    msg = (
        "🏁 *Trade Closed* 🏁\n"
        f"*Symbol:* {trade['symbol']}\n"
        f"*Side:* {trade['side']}\n"
        f"*Entry Price:* {trade['entry_price']:.2f}\n"
        f"*Exit Price:* {exit_price:.2f}\n"
        f"*PNL:* {pnl:.2f}\n"
        f"*Time:* {datetime.utcnow().isoformat()} {emoji}\n"
    )
    return msg


async def shutdown():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
