import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config.settings import global_config, coins_config
from database.db import init_db
from exchange.exchange import create_binance_client
from strategies.ema_cross import monitor_coin
from utils.helpers import shutdown


async def main():
    db = await init_db("data/trades.db")
    binance_config = global_config.get("binance", {})
    binance_client = create_binance_client(
        binance_config.get("apiKey"), binance_config.get("secret")
    )

    tasks = []
    for coin in coins_config:
        tasks.append(
            asyncio.create_task(monitor_coin(coin, global_config, db, binance_client))
        )

    await asyncio.gather(*tasks)
    await db.close()
    await binance_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Завершаем работу...")
        shutdown()
