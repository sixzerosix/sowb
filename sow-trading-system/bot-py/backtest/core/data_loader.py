import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """
    Загружает OHLCV данные. В данном примере - это заглушка,
    генерирующая синтетические 1-минутные данные.

    :param file_path: Путь к файлу (используется для сообщения, но не для загрузки в mock)
    :return: DataFrame с колонками 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    print(f"[LOADER] Загрузка данных (Mock Data) из: {file_path}")

    # 1. Создание тестовых данных (1000 1-минутных свечей)
    np.random.seed(42)  # Для воспроизводимости
    periods = 5000  # Увеличим для более осмысленного бэктеста
    data = {
        "timestamp": pd.to_datetime(
            pd.date_range("2023-01-01", periods=periods, freq="1T")
        ),
        # Генерация "случайного блуждания" для цен
        "close_base": 100 + np.random.randn(periods).cumsum() * 0.1,
        "volume": np.random.randint(100, 1000, periods),
    }
    df = pd.DataFrame(data)

    # Создание OHLC на основе базовой цены
    df["open"] = df["close_base"].shift(1).fillna(df["close_base"][0])
    df["close"] = df["close_base"]
    df["high"] = df[["open", "close"]].max(axis=1) + np.random.rand(periods) * 0.5
    df["low"] = df[["open", "close"]].min(axis=1) - np.random.rand(periods) * 0.5

    # Финальные колонки
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    print(f"[LOADER] Данные успешно загружены. Количество строк: {len(df)}")
    return df
