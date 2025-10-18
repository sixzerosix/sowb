import pandas as pd
import numpy as np
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Пытается загрузить OHLCV данные из CSV по указанному пути,
    используя точные имена колонок, как в предоставленном примере.
    Если файл не загружен или пуст, генерирует синтетические данные (Mock Data).

    :param file_path: Путь к файлу CSV.
    :return: DataFrame с колонками 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    df = pd.DataFrame()
    file_path_obj = Path(file_path)

    # Типы данных, которые мы ожидаем в CSV
    DTYPE_MAPPING = {
        "open": np.float64,
        "high": np.float64,
        "low": np.float64,
        "close": np.float64,
        "volume": np.float64,
    }

    # 1. Попытка загрузить данные из CSV
    if file_path_obj.exists() and file_path_obj.stat().st_size > 0:
        print(f"[LOADER] Попытка загрузить данные из CSV-файла: {file_path}")
        try:
            df = pd.read_csv(
                file_path_obj,
                # Указываем, что первая колонка (timestamp) - это индекс и дата
                index_col="timestamp",
                parse_dates=["timestamp"],
                dtype=DTYPE_MAPPING,
            )

            # Переименовываем индекс в обычную колонку
            df = df.reset_index()

            # Стандартизация названий колонок (перевод в нижний регистр на всякий случай)
            df.columns = [col.lower() for col in df.columns]

            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV-файл не содержит всех необходимых OHLCV колонок.")

            df = df[required_cols]

            # Проверка минимального количества данных для осмысленного бэктеста
            if len(df) < 100:
                print(
                    f"[LOADER] WARNING: В файле '{file_path}' найдено всего {len(df)} строк."
                )
                df = pd.DataFrame()  # Отменяем загрузку

        except Exception as e:
            # Выводим конкретную ошибку, чтобы помочь в отладке
            print(f"[LOADER] ERROR: Ошибка при чтении или преобразовании CSV: {e}")
            df = pd.DataFrame()  # Сбрасываем df, чтобы перейти к Mock Data

    # 2. Если данные не загружены (df.empty == True), генерируем синтетические данные
    if df.empty:
        print(
            f"[LOADER] WARNING: CSV-файл не загружен или пуст. Генерация синтетических данных (Mock Data) для продолжения."
        )

        # Создание тестовых данных (5000 1-минутных свечей)
        np.random.seed(42)  # Для воспроизводимости
        periods = 5000
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
        df["open"] = df["close_base"] + np.random.uniform(-0.01, 0.01, periods)
        df["high"] = df[["open", "close_base"]].max(axis=1) + np.random.uniform(
            0, 0.01, periods
        )
        df["low"] = df[["open", "close_base"]].min(axis=1) - np.random.uniform(
            0, 0.01, periods
        )
        df["close"] = df["close_base"]

        # Очистка и форматирование
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # 3. Финальное сообщение
    print(f"[LOADER] Успешно загружено {len(df)} строк данных. Готов к расчетам.")
    return df
