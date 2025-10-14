import sys
import os
import re
import asyncio
import pandas as pd
from datetime import datetime
from asyncio import WindowsSelectorEventLoopPolicy
import ccxt.async_support as ccxt


async def fetch_historical_data(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date=None,
    limit_per_request=1000,
    max_retries=5,
):
    all_data = []
    since_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else None
    last_timestamp = None
    retries = 0

    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit_per_request,
                since=since_timestamp,
            )

            if not ohlcv:
                break

            # Проверка на дублирующиеся данные
            current_last = ohlcv[-1][0]
            if last_timestamp == current_last:
                break
            last_timestamp = current_last

            # Фильтрация данных по end_timestamp
            if end_timestamp:
                ohlcv = [candle for candle in ohlcv if candle[0] <= end_timestamp]

            all_data.extend(ohlcv)

            # Проверка на завершение сбора данных
            if end_timestamp and current_last >= end_timestamp:
                break

            since_timestamp = ohlcv[-1][0] + 1
            retries = 0  # Сброс счетчика попыток

        except Exception as e:
            print(f"Ошибка: {e}. Попытка {retries}/{max_retries}")
            retries += 1
            if retries > max_retries:
                print("Достигнуто максимальное число попыток. Выход.")
                break
            await asyncio.sleep(5)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


async def save_historical_data(
    exchange_name,
    symbol,
    timeframe,
    start_date,
    end_date=None,
    output_dir="../data/historical",
):
    exchange = getattr(ccxt, exchange_name)(
        {"enableRateLimit": True}
    )  # Встроенный rate limit
    try:
        df = await fetch_historical_data(
            exchange, symbol, timeframe, start_date, end_date
        )
        if df.empty:
            print("Нет данных для сохранения.")
            return
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Безопасное имя файла
        safe_str = lambda s: re.sub(r'[<>:"/\\|?*]', "_", s) if s else ""
        start_str = safe_str(start_date)
        end_str = safe_str(end_date) if end_date else "current"
        file_name = (
            f"{symbol.replace('/', '_')}_{timeframe}_{start_str}_to_{end_str}.csv"
        )

        # Формируем полный путь
        file_name = (
            f"{symbol.replace('/', '_')}_{timeframe}_{start_str}_to_{end_str}.csv"
        )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        full_output_dir = os.path.join(project_root, output_dir)
        full_path = os.path.join(
            full_output_dir, file_name
        )  # Объединяем путь и имя файла

        df.to_csv(full_path, index=False)
        print(f"Данные сохранены в {file_name}")

    finally:
        await exchange.close()  # Гарантированное закрытие соединения


async def main():
    symbols = ["BTC/USDT"]
    timeframes = ["1m"]
    start_date = "2017-01-01 00:00:00"  # YYYY-MM-DD 00:00:00
    end_date = "2017-02-01 00:00:00"
    output_directory = "data/historical"

    # Ограничение одновременных задач
    semaphore = asyncio.Semaphore(2)  # Не более 2 параллельных задач

    async def task_wrapper(symbol, timeframe):
        async with semaphore:
            await save_historical_data(
                "binance",
                symbol,
                timeframe,
                start_date,
                end_date,
                output_dir=output_directory,
            )

    tasks = [
        task_wrapper(symbol, timeframe)
        for symbol in symbols
        for timeframe in timeframes
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # Фикс для Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
