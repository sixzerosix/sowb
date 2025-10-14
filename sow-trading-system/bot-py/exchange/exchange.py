import ccxt.async_support as ccxt
import pandas as pd


def create_bybit_client(apiKey: str = "", secret: str = ""):
    client = ccxt.bybit(
        {
            "enableRateLimit": True,
            "apiKey": apiKey,
            "secret": secret,
        }
    )
    return client


def create_binance_client(apiKey: str = "", secret: str = ""):
    client = ccxt.binance(
        {
            "enableRateLimit": True,
            "apiKey": apiKey,
            "secret": secret,
        }
    )
    return client


async def fetch_data(client, symbol, timeframe, limit):
    try:
        ohlcv = await client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        # Приводим названия колонок к нижнему регистру для унификации
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"Ошибка получения данных с {client.id}: {e}")
        return None
