import time
import pandas as pd
import numpy as np
import copy
from typing import Tuple, Dict, Any, Optional, List
import optuna
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from backtest.core.data_loader import load_data
from backtest.core.config import FILE_PATH

# Инициализация консоли Rich
console = Console()

# =========================================================================
# === 1. КОНФИГУРАЦИЯ (StrategyConfig, PersistenceConfig, OptimizationParams) ===
# =========================================================================


@dataclass
class StrategyConfig:
    """
    Конфигурация торговой стратегии и параметров индикаторов.
    """

    initial_capital: float = 1000.0
    leverage: float = 1.0  # Плечо (для упрощения в бэктесте не используется)
    target_roi_percent: float = 1.0  # Целевой TP % от позиции
    risk_roi_percent: float = 0.5  # Целевой SL % от позиции
    # Словарь для динамической настройки индикаторов
    indicator_set: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PersistenceConfig:
    """
    Конфигурация сохранения результатов.
    """

    save_to_sqlite: bool = False
    save_to_csv: bool = False
    save_to_txt: bool = False
    sqlite_db_name: str = "backtest_results.db"
    optimization_table_name: str = "optimization_results"
    output_file_prefix: str = "trades_"


class OptimizationParams:
    # Параметры поиска Optuna
    EMA_FAST_RANGE: Tuple[int, int] = (5, 50)
    EMA_SLOW_RANGE: Tuple[int, int] = (20, 150)
    RSI_LEN_RANGE: Tuple[int, int] = (7, 30)
    RSI_ENTRY_RANGE: Tuple[int, int] = (
        25,
        45,
    )  # Уровень перепроданности для LONG входа
    RSI_EXIT_RANGE: Tuple[int, int] = (
        55,
        75,
    )  # Уровень перекупленности для SHORT входа
    ATR_LEN_RANGE: Tuple[int, int] = (5, 30)
    N_TRIALS: int = 100  # Увеличено для более качественной оптимизации
    OPTIMIZATION_METRIC: str = "Total PnL"


DEFAULT_CONFIG = StrategyConfig(
    initial_capital=1000.0,
    indicator_set={
        "EMA_TREND": {"fast_len": 9, "slow_len": 21},
        "ATR_STOP": {"atr_len": 14, "multiplier": 1.5},
        "RSI_ENTRY_EXIT": {"rsi_len": 14, "entry_low": 30, "exit_high": 70},
    },
)
STRATEGY_CONFIGS = {"DEFAULT_CONFIG": DEFAULT_CONFIG}


# =========================================================================
# === 2. ЗАГРУЗКА ДАННЫХ (load_data) ===
# =========================================================================


# АКТИВИРОВАНА заглушка для загрузки данных, чтобы сделать файл самодостаточным
# def load_data() -> pd.DataFrame:
#     """
#     Генерация более реалистичных синтетических данных OHLCV.
#     В реальном проекте здесь будет код для загрузки файла CSV.
#     """
#     console.print(
#         f"[bold blue][LOADER][/bold blue] Загрузка синтетических данных (Mock/Real CSV Placeholder)..."
#     )
#     periods = 5000  # Увеличено для более осмысленного бэктеста
#     np.random.seed(42)

#     # 1. Генерация базовой цены (случайное блуждание с трендом)
#     base_price = 100 + np.random.randn(periods).cumsum() * 0.1
#     # Добавление слабого восходящего тренда
#     trend = np.linspace(0, 5, periods)
#     close_price = base_price + trend

#     # 2. Создание OHLCV
#     data = {
#         "timestamp": pd.date_range("2025-07-01", periods=periods, freq="1T"),
#         "close": close_price,
#     }
#     df = pd.DataFrame(data)

#     # Open = Close shift(1) + небольшая случайная вариация
#     df["open"] = df["close"].shift(1).fillna(df["close"]) * (
#         1 + np.random.uniform(-0.0001, 0.0001, periods)
#     )

#     # High/Low вокруг Open и Close
#     df["high"] = df[["open", "close"]].max(axis=1) + np.random.rand(periods) * 0.05
#     df["low"] = df[["open", "close"]].min(axis=1) - np.random.rand(periods) * 0.05

#     # Volume
#     df["volume"] = np.random.randint(100, 10000, periods)

#     df = df.dropna().set_index("timestamp")

#     console.print(
#         f"[bold blue][LOADER][/bold blue] Загружено {len(df)} строк данных с {df.index.min()} по {df.index.max()}."
#     )
#     return df


# =========================================================================
# === 3. ИНДИКАТОРЫ (calculate_strategy_indicators - БЕЗ TALIB) ===
# =========================================================================


# Вспомогательная функция для расчета RSI (без talib)
def _calculate_rsi(series: pd.Series, length: int) -> pd.Series:
    """Расчет RSI, используя чистый Pandas/Numpy."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Средние значения (SMA/RMA) - используем EMA для сглаживания, как и в TALIB (RMA)
    avg_gain = gain.ewm(span=length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(span=length, adjust=False, min_periods=length).mean()

    # Избегаем деления на ноль
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Вспомогательная функция для расчета ATR (без talib)
def _calculate_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Расчет ATR, используя чистый Pandas/Numpy."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    # True Range (TR)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Average True Range (ATR) - используем RMA (EMA) для сглаживания
    atr = tr.ewm(span=length, adjust=False, min_periods=length).mean()
    return atr


def calculate_strategy_indicators(
    df: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """
    Расчет технических индикаторов на основе конфигурации StrategyConfig.indicator_set.
    Использует чистые функции Pandas/Numpy.
    """
    df = df.copy()
    params = config.indicator_set

    # 1. EMA Trend
    ema_params = params.get("EMA_TREND", {})
    fast_len = ema_params.get("fast_len", 9)
    slow_len = ema_params.get("slow_len", 21)

    df["ema_fast"] = df["close"].ewm(span=fast_len, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_len, adjust=False).mean()
    df["trend_up"] = df["ema_fast"] > df["ema_slow"]
    df["trend_down"] = df["ema_fast"] < df["ema_slow"]  # Добавлено для SHORT

    # 2. RSI
    rsi_params = params.get("RSI_ENTRY_EXIT", {})
    rsi_len = rsi_params.get("rsi_len", 14)
    df["rsi_val"] = _calculate_rsi(df["close"], rsi_len)

    # 3. ATR
    atr_params = params.get("ATR_STOP", {})
    atr_len = atr_params.get("atr_len", 14)
    df["atr_val"] = _calculate_atr(df, atr_len)

    # Удаляем NaN после всех расчетов
    df_clean = df.dropna()

    console.print(
        f"[bold yellow][INDICATORS][/bold yellow] Рассчитано. Исходно: {len(df)}, После очистки: {len(df_clean)}"
    )
    return df_clean


# =========================================================================
# === 4. BACKTEST ENGINE (backtest_engine) ===
# =========================================================================


def backtest_engine(
    df: pd.DataFrame, config: StrategyConfig
) -> Tuple[pd.DataFrame, float]:
    """
    Реалистичный (но упрощенный) Backtest Engine.
    Отслеживает открытые сделки, использует ATR для SL и RSI для TP/фильтра.
    """

    initial_capital = config.initial_capital

    # Параметры стратегии
    rsi_entry_low = config.indicator_set.get("RSI_ENTRY_EXIT", {}).get("entry_low", 30)
    rsi_exit_high = config.indicator_set.get("RSI_ENTRY_EXIT", {}).get("exit_high", 70)
    atr_multiplier = config.indicator_set.get("ATR_STOP", {}).get("multiplier", 1.5)

    trades = []
    equity_history: List[float] = [initial_capital]

    current_equity = initial_capital

    # Структура открытой позиции: (entry_index, entry_price, side, stop_loss_price, take_profit_price)
    open_position: Optional[Tuple[int, float, str, float, float]] = None
    trade_id = 0

    # Проходим по данным, начиная с момента, когда есть индикаторы
    for i in range(1, len(df)):
        current_row = df.iloc[i]

        # ---------------------------------
        # 1. ОБРАБОТКА ОТКРЫТОЙ ПОЗИЦИИ
        # ---------------------------------
        if open_position is not None:
            entry_index, entry_price, side, sl_price, tp_price = open_position
            pnl = 0.0

            # --- Условия выхода (SL/TP) ---
            exit_reason = None
            exit_price = current_row["close"]  # Цена закрытия по умолчанию

            if side == "LONG":
                # Exit by Stop Loss (Low < SL)
                if current_row["low"] <= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                # Exit by Take Profit (High >= TP)
                elif current_row["high"] >= tp_price:
                    exit_reason = "TP"
                    exit_price = tp_price
                # Exit by reverse trend (RSI too high)
                elif current_row["rsi_val"] >= rsi_exit_high:
                    exit_reason = "RSI_EXIT"
                    exit_price = current_row["close"]

            elif side == "SHORT":
                # Exit by Stop Loss (High >= SL)
                if current_row["high"] >= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                # Exit by Take Profit (Low <= TP)
                elif current_row["low"] <= tp_price:
                    exit_reason = "TP"
                    exit_price = tp_price
                # Exit by reverse trend (RSI too low)
                elif current_row["rsi_val"] <= rsi_entry_low:
                    exit_reason = "RSI_EXIT"
                    exit_price = current_row["close"]

            # Если позиция закрыта (СКОРРЕКТИРОВАННЫЙ РАСЧЕТ PNL)
            if exit_reason is not None:
                # PnL рассчитывается как процентное изменение, умноженное на капитал (упрощенно)
                if side == "LONG":
                    pnl = (exit_price - entry_price) / entry_price * current_equity
                elif side == "SHORT":
                    pnl = (entry_price - exit_price) / entry_price * current_equity
                else:
                    pnl = 0.0

                pnl_percent = pnl / current_equity  # Процент PnL от текущего капитала

                # Обновление капитала
                current_equity += pnl

                # Запись сделки
                is_liquidation = (
                    1 if exit_reason == "SL" else 0
                )  # Упрощенно: SL считается "ликвидацией"
                trades.append(
                    {
                        "trade_id": trade_id,
                        "entry_time": df.index[entry_index],
                        "exit_time": current_row.name,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_percent": pnl_percent * 100,
                        "entry_value": initial_capital,
                        "liquidation": is_liquidation,
                    }
                )
                open_position = None  # Закрываем позицию

        # ---------------------------------
        # 2. УСЛОВИЯ ВХОДА
        # ---------------------------------
        if open_position is None:

            # --- LONG ENTRY ---
            long_entry_condition = (
                current_row["trend_up"] == True  # EMA fast > EMA slow
                and current_row["rsi_val"] < rsi_entry_low  # RSI в зоне перепроданности
            )

            if long_entry_condition:
                trade_id += 1

                entry_price = current_row["close"]
                current_atr = current_row["atr_val"]

                # Расчет SL/TP на основе ATR
                sl_amount = current_atr * atr_multiplier
                sl_price = entry_price - sl_amount

                # Для простоты: TP - это фиксированное ATR расстояние (2x риск)
                tp_price = entry_price + sl_amount * 2

                open_position = (i, entry_price, "LONG", sl_price, tp_price)

            # --- SHORT ENTRY (ДОБАВЛЕНО) ---
            short_entry_condition = (
                current_row["trend_down"] == True  # EMA fast < EMA slow
                and current_row["rsi_val"] > rsi_exit_high  # RSI в зоне перекупленности
            )

            if short_entry_condition:
                trade_id += 1

                entry_price = current_row["close"]
                current_atr = current_row["atr_val"]

                # Расчет SL/TP на основе ATR
                sl_amount = current_atr * atr_multiplier
                sl_price = entry_price + sl_amount

                # Для простоты: TP - это фиксированное ATR расстояние (2x риск)
                tp_price = entry_price - sl_amount * 2

                open_position = (i, entry_price, "SHORT", sl_price, tp_price)

        equity_history.append(current_equity)

    trades_df = pd.DataFrame(trades)

    console.print(
        f"[bold green][BACKTEST][/bold green] Завершено. Сделок: {len(trades_df)}, Финальный капитал: {current_equity:.2f}"
    )
    return trades_df, current_equity


# =========================================================================
# === 5. АНАЛИЗ И ОТЧЕТНОСТЬ (analysis & persistence) ===
# =========================================================================


def calculate_metrics(
    trades_df: pd.DataFrame, initial_capital: float, final_equity: float
) -> Tuple[Dict[str, Any], pd.Series, pd.Series]:
    """Рассчитывает ключевые метрики бэктеста."""
    if trades_df.empty:
        # Если сделок нет, возвращаем пустые метрики
        return (
            {"Total PnL": -100.0, "Sharpe Ratio": -1.0, "Max Drawdown (%)": 100.0},
            pd.Series([initial_capital]),
            pd.Series([0.0]),
        )

    # Добавляем колонку 'liquidation' для совместимости, если ее нет (Mock)
    if "liquidation" not in trades_df.columns:
        trades_df["liquidation"] = 0

    total_pnl = trades_df["pnl"].sum()
    success_rate = (trades_df["pnl"] > 0).mean() * 100
    liquidation_rate = trades_df["liquidation"].mean() * 100
    num_trades = len(trades_df)

    return_on_capital = (final_equity - initial_capital) / initial_capital * 100

    # Расчет Максимальной Просадки (Max Drawdown - MDD)
    equity = initial_capital + trades_df["pnl"].cumsum()
    # Добавляем начальный капитал
    equity_with_start = pd.concat([pd.Series([initial_capital]), equity])

    peak = equity_with_start.cummax()
    drawdown = peak - equity_with_start
    max_drawdown = drawdown.max() / initial_capital * 100
    max_drawdown_amount = drawdown.max()
    drawdown_for_plot = (
        drawdown / initial_capital
    )  # Нормализованный drawdown для графика

    # Расчет Sharpe Ratio (упрощенно)
    returns = trades_df["pnl"] / trades_df["entry_value"]  # Нормализованный PnL
    sharpe_ratio = (
        (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)
        if returns.std() != 0
        else 0.0
    )

    metrics = {
        "Initial Capital": initial_capital,
        "Final Equity": final_equity,
        "Total PnL": total_pnl,
        "RoC (%)": return_on_capital,
        "Num Trades": num_trades,
        "Success Rate (%)": success_rate,
        "Liquidation Rate (%)": liquidation_rate,
        "Max Drawdown (%)": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
    }

    return metrics, equity_with_start, drawdown_for_plot


def display_results_rich(
    metrics: Dict[str, Any],
    trades_df: pd.DataFrame,
    execution_time: float,
    config: StrategyConfig,
):
    """Отображает результаты в красивом формате Rich."""

    # 1. Обзорная таблица метрик
    table = Table(
        title="[bold green]Ключевые Метрики Бэктеста[/bold green]",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Метрика", style="dim", justify="left")
    table.add_column("Значение", style="yellow", justify="right")

    pnl_style = "bold green" if metrics.get("Total PnL", 0) >= 0 else "bold red"
    dd_style = "bold red" if metrics.get("Max Drawdown (%)", 0) > 10 else "bold green"

    table.add_row("Начальный капитал", f"${metrics['Initial Capital']:.2f}")
    table.add_row("Финальный капитал", f"${metrics['Final Equity']:.2f}")
    table.add_row(
        "Общий PnL", f"[{pnl_style}]${metrics['Total PnL']:.2f}[/{pnl_style}]"
    )
    table.add_row(
        "Доходность (RoC)", f"[{pnl_style}]{metrics['RoC (%)']:.2f}%[/{pnl_style}]"
    )
    table.add_row(
        "Max Просадка (MDD)",
        f"[{dd_style}]{metrics['Max Drawdown (%)']:.2f}%[/{dd_style}]",
    )
    table.add_row("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.4f}")
    table.add_row("Количество сделок", f"{metrics['Num Trades']:,}")
    table.add_row("Процент успеха", f"{metrics['Success Rate (%)']:.2f}%")

    console.print(Panel(table))

    # 2. Конфигурация
    config_table = Table(
        title="[bold yellow]Оптимизированная Конфигурация[/bold yellow]",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )
    config_table.add_column("Параметр", style="dim")
    config_table.add_column("Значение", style="yellow")

    ema = config.indicator_set.get("EMA_TREND", {})
    rsi = config.indicator_set.get("RSI_ENTRY_EXIT", {})
    atr = config.indicator_set.get("ATR_STOP", {})

    config_table.add_row("EMA Fast Len", str(ema.get("fast_len")))
    config_table.add_row("EMA Slow Len", str(ema.get("slow_len")))
    config_table.add_row("RSI Len", str(rsi.get("rsi_len")))
    config_table.add_row("RSI Entry Low", str(rsi.get("entry_low")))
    config_table.add_row("RSI Exit High", str(rsi.get("exit_high")))
    config_table.add_row("ATR Len", str(atr.get("atr_len")))
    config_table.add_row("ATR Multiplier (SL/TP)", str(atr.get("multiplier")))

    console.print(Panel(config_table))

    console.print(
        f"\n[bold white]Время выполнения:[/bold white] {execution_time:.2f} сек."
    )


def plot_results(
    equity_for_plot: pd.Series,
    drawdown_for_plot: pd.Series,
    trades_df: pd.DataFrame,
    initial_capital: float,
    metric_name: str,
    best_metrics: Dict[str, Any],
):
    """Строит график кривой эквити и просадки."""
    try:
        if equity_for_plot.empty:
            console.print("[PLOT] Нет данных для построения графика.")
            return

        # Индекс для графика (количество сделок)
        trade_indices = np.arange(len(equity_for_plot))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Backtest Results (Optimized for {metric_name})", fontsize=16)

        # График Эквити
        ax1.plot(
            trade_indices, equity_for_plot.values, label="Equity Curve", color="green"
        )
        ax1.axhline(
            initial_capital,
            color="grey",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )
        ax1.set_title("Кривая Эквити (Equity Curve)", fontsize=14)
        ax1.set_ylabel("Капитал (USD)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # График Просадки
        # drawdown_for_plot уже нормализован и включает 0.0 в начале
        ax2.fill_between(
            trade_indices, 0, drawdown_for_plot.values * 100, color="red", alpha=0.4
        )
        ax2.plot(
            trade_indices,
            drawdown_for_plot.values * 100,
            color="red",
            label="Drawdown (%)",
        )
        ax2.set_title(
            f"Относительная Просадка (Max DD: {best_metrics['Max Drawdown (%)']:.2f}%)",
            fontsize=14,
        )
        ax2.set_ylabel("Просадка (%)", fontsize=12)
        ax2.set_xlabel("Номер закрытой сделки", fontsize=12)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Учитываем заголовок
        plt.show()

    except ImportError:
        console.print(
            "[bold red][PLOT ERROR][/bold red] Не удалось построить график. Установите matplotlib: [yellow]pip install matplotlib[/yellow]"
        )
    except Exception as e:
        console.print(
            f"[bold red][PLOT ERROR][/bold red] Произошла ошибка при построении графика: {e}"
        )


def persist_optimization_result(
    best_config: StrategyConfig,
    best_metrics: Dict[str, Any],
    persistence_config: PersistenceConfig,
):
    """
    Сохранение лучшего результата оптимизации.
    В этой объединенной версии используется только Mock-сохранение.
    """

    # 1. Подготовка записи
    record = {
        "timestamp": pd.Timestamp.now(),
        "total_pnl": best_metrics.get("Total PnL", 0),
        "sharpe_ratio": best_metrics.get("Sharpe Ratio", 0),
        "max_drawdown_percent": best_metrics.get("Max Drawdown (%)", 0),
        "num_trades": best_metrics.get("Num Trades", 0),
        # Параметры индикаторов
        "ema_fast_len": best_config.indicator_set.get("EMA_TREND", {}).get("fast_len"),
        "ema_slow_len": best_config.indicator_set.get("EMA_TREND", {}).get("slow_len"),
        "rsi_len": best_config.indicator_set.get("RSI_ENTRY_EXIT", {}).get("rsi_len"),
        "rsi_entry_low": best_config.indicator_set.get("RSI_ENTRY_EXIT", {}).get(
            "entry_low"
        ),
        "atr_len": best_config.indicator_set.get("ATR_STOP", {}).get("atr_len"),
        "atr_multiplier": best_config.indicator_set.get("ATR_STOP", {}).get(
            "multiplier"
        ),
    }

    results_df = pd.DataFrame([record])

    console.print(
        Panel(
            f"[bold magenta][PERSIST] Сохранение лучшего результата (Mock).[/bold magenta]\n"
            f"В реальном проекте здесь будет код для сохранения в SQLite/CSV.",
            title="Сохранение",
            border_style="magenta",
        )
    )


# =========================================================================
# === 6. ЛОГИКА ОПТИМИЗАЦИИ OPTUNA (ОСНОВНАЯ ЧАСТЬ) ===
# =========================================================================


def objective(
    trial: optuna.Trial, data_df: pd.DataFrame, base_config: StrategyConfig
) -> float:
    """
    Целевая функция для Optuna.
    """
    config = copy.deepcopy(base_config)

    # 1. Определение пространства поиска
    fast_len = trial.suggest_int(
        "ema_fast_len",
        OptimizationParams.EMA_FAST_RANGE[0],
        OptimizationParams.EMA_FAST_RANGE[1],
    )
    slow_len = trial.suggest_int(
        "ema_slow_len",
        max(fast_len + 1, OptimizationParams.EMA_SLOW_RANGE[0]),
        OptimizationParams.EMA_SLOW_RANGE[1],
    )

    rsi_len = trial.suggest_int(
        "rsi_len",
        OptimizationParams.RSI_LEN_RANGE[0],
        OptimizationParams.RSI_LEN_RANGE[1],
    )
    rsi_entry_low = trial.suggest_int(
        "rsi_entry_low",
        OptimizationParams.RSI_ENTRY_RANGE[0],
        OptimizationParams.RSI_ENTRY_RANGE[1],
    )
    rsi_exit_high = trial.suggest_int(
        "rsi_exit_high",
        OptimizationParams.RSI_EXIT_RANGE[0],
        OptimizationParams.RSI_EXIT_RANGE[1],
    )

    if rsi_entry_low >= rsi_exit_high:
        raise optuna.exceptions.TrialPruned(
            "RSI Entry level must be lower than RSI Exit level."
        )

    atr_len = trial.suggest_int(
        "atr_len",
        OptimizationParams.ATR_LEN_RANGE[0],
        OptimizationParams.ATR_LEN_RANGE[1],
    )

    # 2. Применяем выбранные параметры к конфигурации
    config.indicator_set["EMA_TREND"] = {"fast_len": fast_len, "slow_len": slow_len}
    config.indicator_set["RSI_ENTRY_EXIT"] = {
        "rsi_len": rsi_len,
        "entry_low": rsi_entry_low,
        "exit_high": rsi_exit_high,
    }
    config.indicator_set["ATR_STOP"] = {
        "atr_len": atr_len,
        "multiplier": 1.5,
    }  # multiplier не оптимизируем для простоты

    # 3. Расчет индикаторов и запуск бэктеста
    df_with_indicators = calculate_strategy_indicators(data_df, config)
    trades_df, final_equity = backtest_engine(df_with_indicators, config)

    # 4. Расчет метрик и возврат целевого значения
    if trades_df.empty:
        return -100.0  # Низкий штраф

    metrics, _, _ = calculate_metrics(
        trades_df, base_config.initial_capital, final_equity
    )

    metric_name = OptimizationParams.OPTIMIZATION_METRIC

    if metric_name not in metrics:
        return -100.0

    # Optuna максимизирует, поэтому для минимизации DD возвращаем отрицательное значение
    if metric_name == "Max Drawdown (%)":
        return -metrics[metric_name]

    return metrics[metric_name]


def optimize_strategy_optuna(
    data_df: pd.DataFrame,
    base_config: StrategyConfig,
    persistence_config: PersistenceConfig,
    start_time: float,
) -> Optional[StrategyConfig]:
    """
    Основная функция для запуска оптимизации с помощью Optuna.
    """

    metric_name = OptimizationParams.OPTIMIZATION_METRIC
    direction = "minimize" if metric_name == "Max Drawdown (%)" else "maximize"

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),  # Для воспроизводимости
    )

    console.print(f"\n[bold yellow]--- НАЧАЛО ОПТИМИЗАЦИИ OPTUNA ---[/bold yellow]")
    console.print(
        f"Целевая метрика: [cyan]{metric_name}[/cyan] ([magenta]{direction}[/magenta])"
    )
    console.print(f"Количество попыток: {OptimizationParams.N_TRIALS}")

    study.optimize(
        lambda trial: objective(trial, data_df, base_config),
        n_trials=OptimizationParams.N_TRIALS,
        n_jobs=-1,  # Параллельный запуск
        show_progress_bar=True,
    )

    console.print("\n[bold yellow]--- РЕЗУЛЬТАТЫ OPTUNA ---[/bold yellow]")

    best_trial = study.best_trial

    if best_trial.value <= -99.0:
        console.print(
            "[bold red][OPTIMIZATION][/bold red] Optuna не нашла ни одной конфигурации, совершившей сделки."
        )
        return None

    console.print(
        f"Лучшее значение целевой метрики ({metric_name}): [green]{best_trial.value:.2f}[/green]"
    )
    console.print(f"Лучшие параметры: [yellow]{best_trial.params}[/yellow]")

    # Формируем лучшую конфигурацию StrategyConfig
    best_config = copy.deepcopy(base_config)
    best_params = best_trial.params

    # Записываем лучший набор параметров в конфигурацию
    best_config.indicator_set["EMA_TREND"] = {
        "fast_len": best_params.get("ema_fast_len", 9),
        "slow_len": best_params.get("ema_slow_len", 21),
    }
    best_config.indicator_set["RSI_ENTRY_EXIT"] = {
        "rsi_len": best_params.get("rsi_len", 14),
        "entry_low": best_params.get("rsi_entry_low", 30),
        "exit_high": best_params.get("rsi_exit_high", 70),
    }
    best_config.indicator_set["ATR_STOP"] = {
        "atr_len": best_params.get("atr_len", 14),
        "multiplier": 1.5,
    }

    # --- ФИНАЛЬНЫЙ ПРОГОН ЛУЧШЕЙ КОНФИГУРАЦИИ ---
    df_with_indicators = calculate_strategy_indicators(data_df, best_config)
    trades_df_final, final_equity_final = backtest_engine(
        df_with_indicators, best_config
    )

    if trades_df_final.empty:
        console.print(
            "[bold red][FINAL RUN][/bold red] Лучшая конфигурация не совершила сделок. Пропускаем сохранение."
        )
        return None

    best_metrics, equity_for_plot, drawdown_for_plot = calculate_metrics(
        trades_df_final, base_config.initial_capital, final_equity_final
    )

    execution_time_opt = time.time() - start_time

    # Сохранение результатов и отображение
    persist_optimization_result(best_config, best_metrics, persistence_config)
    display_results_rich(best_metrics, trades_df_final, execution_time_opt, best_config)

    # График
    plot_results(
        equity_for_plot,
        drawdown_for_plot,
        trades_df_final,
        best_config.initial_capital,
        metric_name,
        best_metrics,
    )

    return best_config


# =========================================================================
# === ОСНОВНАЯ ТОЧКА ВХОДА ===
# =========================================================================

if __name__ == "__main__":

    # ПРИМЕЧАНИЕ: Optuna использует multiprocessing, что может вызвать ошибки
    # при запуске в не-основном процессе. Если возникнет ошибка, попробуйте
    # обернуть main-логику в 'if __name__ == "__main__":'

    start_time = time.time()

    # 1. Загрузка данных (Используется генератор)
    data_df = load_data(str(FILE_PATH))
    # data_df = load_data()

    # 2. Определение конфигурации
    target_config = STRATEGY_CONFIGS.get("DEFAULT_CONFIG", DEFAULT_CONFIG)
    persistence_config = PersistenceConfig(save_to_csv=True)

    RUN_MODE = "OPTIMIZE"  # Запускаем в режиме оптимизации

    if RUN_MODE == "OPTIMIZE":

        best_config = optimize_strategy_optuna(
            data_df, target_config, persistence_config, start_time
        )

        execution_time_opt = time.time() - start_time

        if best_config:
            # Результаты уже выведены в display_results_rich
            console.print(
                "\n[bold green]--- ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО ---[/bold green]"
            )
            console.print(f"Общее время выполнения: {execution_time_opt:.2f} сек.")

        else:
            console.print("\n[bold red]--- ОШИБКА ОПТИМИЗАЦИИ ---[/bold red]")
            console.print(
                "[bold red]Не удалось найти лучшую конфигурацию.[/bold red] Проверьте, совершила ли стратегия хотя бы одну сделку при заданных диапазонах параметров."
            )
