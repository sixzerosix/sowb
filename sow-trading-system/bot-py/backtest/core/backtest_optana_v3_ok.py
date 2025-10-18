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
from numba import jit, prange

from backtest.core.data_loader import load_data
from backtest.core.config import FILE_PATH

# Инициализация консоли Rich
console = Console()

# ================================================================================
# === 1. КОНФИГУРАЦИЯ (StrategyConfig, PersistenceConfig, OptimizationParams) ===
# ================================================================================


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


# ================================================================================
# === 2. NUMBA-УСКОРЕННЫЕ ИНДИКАТОРЫ ===
# ================================================================================


@jit(nopython=True)
def calculate_ema_numba(data: np.ndarray, span: int) -> np.ndarray:
    """Расчет EMA с использованием Numba."""
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(data)
    result[0] = data[0]

    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


@jit(nopython=True)
def calculate_rsi_numba(data: np.ndarray, length: int) -> np.ndarray:
    """Расчет RSI с использованием Numba."""
    n = len(data)
    rsi = np.full(n, np.nan)

    if n < length + 1:
        return rsi

    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Первое среднее
    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])

    if avg_loss == 0:
        rsi[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[length] = 100.0 - (100.0 / (1.0 + rs))

    # EMA для остальных значений
    alpha = 1.0 / length
    for i in range(length, n - 1):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@jit(nopython=True)
def calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int
) -> np.ndarray:
    """Расчет ATR с использованием Numba."""
    n = len(high)
    atr = np.full(n, np.nan)

    if n < length + 1:
        return atr

    # True Range
    tr = np.empty(n - 1)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i - 1] = max(hl, hc, lc)

    # Первое среднее
    atr[length] = np.mean(tr[:length])

    # EMA для остальных значений
    alpha = 1.0 / length
    for i in range(length, n - 1):
        atr[i + 1] = alpha * tr[i] + (1 - alpha) * atr[i]

    return atr


def calculate_strategy_indicators(
    df: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """
    Расчет технических индикаторов с использованием Numba.
    """
    df = df.copy()
    params = config.indicator_set

    # Преобразуем данные в numpy массивы
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values

    # 1. EMA Trend
    ema_params = params.get("EMA_TREND", {})
    fast_len = ema_params.get("fast_len", 9)
    slow_len = ema_params.get("slow_len", 21)

    df["ema_fast"] = calculate_ema_numba(close_arr, fast_len)
    df["ema_slow"] = calculate_ema_numba(close_arr, slow_len)
    df["trend_up"] = df["ema_fast"] > df["ema_slow"]
    df["trend_down"] = df["ema_fast"] < df["ema_slow"]

    # 2. RSI
    rsi_params = params.get("RSI_ENTRY_EXIT", {})
    rsi_len = rsi_params.get("rsi_len", 14)
    df["rsi_val"] = calculate_rsi_numba(close_arr, rsi_len)

    # 3. ATR
    atr_params = params.get("ATR_STOP", {})
    atr_len = atr_params.get("atr_len", 14)
    df["atr_val"] = calculate_atr_numba(high_arr, low_arr, close_arr, atr_len)

    # Удаляем NaN после всех расчетов
    df_clean = df.dropna()

    console.print(
        f"[bold yellow][INDICATORS][/bold yellow] Рассчитано (Numba). Исходно: {len(df)}, После очистки: {len(df_clean)}"
    )
    return df_clean


# ================================================================================
# === 3. NUMBA-УСКОРЕННЫЙ BACKTEST ENGINE ===
# ================================================================================


@jit(nopython=True)
def backtest_core_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    trend_up: np.ndarray,
    trend_down: np.ndarray,
    rsi_val: np.ndarray,
    atr_val: np.ndarray,
    initial_capital: float,
    rsi_entry_low: float,
    rsi_exit_high: float,
    atr_multiplier: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Ядро бэктеста с использованием Numba.
    Возвращает массивы данных о сделках.
    """
    n = len(close)
    max_trades = n  # Максимальное количество сделок

    # Массивы для хранения сделок
    entry_indices = np.empty(max_trades, dtype=np.int32)
    exit_indices = np.empty(max_trades, dtype=np.int32)
    entry_prices = np.empty(max_trades, dtype=np.float64)
    exit_prices = np.empty(max_trades, dtype=np.float64)
    pnls = np.empty(max_trades, dtype=np.float64)
    sides = np.empty(max_trades, dtype=np.int8)  # 1 = LONG, -1 = SHORT
    exit_reasons = np.empty(max_trades, dtype=np.int8)  # 0 = SL, 1 = TP, 2 = RSI_EXIT
    liquidations = np.empty(max_trades, dtype=np.int8)

    trade_count = 0
    current_equity = initial_capital

    # Состояние позиции: 0 = нет позиции, 1 = LONG, -1 = SHORT
    position_side = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_idx = 0

    for i in range(1, n):
        # Проверка открытой позиции
        if position_side != 0:
            exit_reason = -1
            exit_price = close[i]

            if position_side == 1:  # LONG
                if low[i] <= sl_price:
                    exit_reason = 0  # SL
                    exit_price = sl_price
                elif high[i] >= tp_price:
                    exit_reason = 1  # TP
                    exit_price = tp_price
                elif rsi_val[i] >= rsi_exit_high:
                    exit_reason = 2  # RSI_EXIT
                    exit_price = close[i]

            elif position_side == -1:  # SHORT
                if high[i] >= sl_price:
                    exit_reason = 0  # SL
                    exit_price = sl_price
                elif low[i] <= tp_price:
                    exit_reason = 1  # TP
                    exit_price = tp_price
                elif rsi_val[i] <= rsi_entry_low:
                    exit_reason = 2  # RSI_EXIT
                    exit_price = close[i]

            # Закрытие позиции
            if exit_reason >= 0:
                if position_side == 1:
                    pnl = (exit_price - entry_price) / entry_price * current_equity
                else:
                    pnl = (entry_price - exit_price) / entry_price * current_equity

                current_equity += pnl

                # Сохранение сделки
                entry_indices[trade_count] = entry_idx
                exit_indices[trade_count] = i
                entry_prices[trade_count] = entry_price
                exit_prices[trade_count] = exit_price
                pnls[trade_count] = pnl
                sides[trade_count] = position_side
                exit_reasons[trade_count] = exit_reason
                liquidations[trade_count] = 1 if exit_reason == 0 else 0

                trade_count += 1
                position_side = 0

        # Проверка условий входа
        if position_side == 0:
            # LONG ENTRY
            if trend_up[i] and rsi_val[i] < rsi_entry_low:
                entry_price = close[i]
                sl_amount = atr_val[i] * atr_multiplier
                sl_price = entry_price - sl_amount
                tp_price = entry_price + sl_amount * 2
                position_side = 1
                entry_idx = i

            # SHORT ENTRY
            elif trend_down[i] and rsi_val[i] > rsi_exit_high:
                entry_price = close[i]
                sl_amount = atr_val[i] * atr_multiplier
                sl_price = entry_price + sl_amount
                tp_price = entry_price - sl_amount * 2
                position_side = -1
                entry_idx = i

    return (
        entry_indices[:trade_count],
        exit_indices[:trade_count],
        entry_prices[:trade_count],
        exit_prices[:trade_count],
        pnls[:trade_count],
        sides[:trade_count],
        exit_reasons[:trade_count],
        liquidations[:trade_count],
        trade_count,
    )


def backtest_engine(
    df: pd.DataFrame, config: StrategyConfig
) -> Tuple[pd.DataFrame, float]:
    """
    Обертка для Numba-ускоренного бэктеста.
    """
    initial_capital = config.initial_capital

    # Параметры стратегии
    rsi_entry_low = config.indicator_set.get("RSI_ENTRY_EXIT", {}).get("entry_low", 30)
    rsi_exit_high = config.indicator_set.get("RSI_ENTRY_EXIT", {}).get("exit_high", 70)
    atr_multiplier = config.indicator_set.get("ATR_STOP", {}).get("multiplier", 1.5)

    # Преобразование данных в numpy массивы
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    trend_up_arr = df["trend_up"].values
    trend_down_arr = df["trend_down"].values
    rsi_arr = df["rsi_val"].values
    atr_arr = df["atr_val"].values

    # Запуск Numba-ускоренного бэктеста
    (
        entry_indices,
        exit_indices,
        entry_prices,
        exit_prices,
        pnls,
        sides,
        exit_reasons,
        liquidations,
        trade_count,
    ) = backtest_core_numba(
        close_arr,
        high_arr,
        low_arr,
        trend_up_arr,
        trend_down_arr,
        rsi_arr,
        atr_arr,
        initial_capital,
        rsi_entry_low,
        rsi_exit_high,
        atr_multiplier,
    )

    # Формирование DataFrame с результатами
    trades = []
    current_equity = initial_capital

    for i in range(trade_count):
        current_equity_before = current_equity
        current_equity += pnls[i]

        side_str = "LONG" if sides[i] == 1 else "SHORT"

        trades.append(
            {
                "trade_id": i,
                "entry_time": df.index[entry_indices[i]],
                "exit_time": df.index[exit_indices[i]],
                "side": side_str,
                "entry_price": entry_prices[i],
                "exit_price": exit_prices[i],
                "pnl": pnls[i],
                "pnl_percent": (pnls[i] / current_equity_before) * 100,
                "entry_value": initial_capital,
                "liquidation": liquidations[i],
            }
        )

    trades_df = pd.DataFrame(trades)
    final_equity = initial_capital + pnls.sum() if trade_count > 0 else initial_capital

    console.print(
        f"[bold green][BACKTEST][/bold green] Завершено (Numba). Сделок: {len(trades_df)}, Финальный капитал: {final_equity:.2f}"
    )
    return trades_df, final_equity


# ================================================================================
# === 4. АНАЛИЗ И ОТЧЕТНОСТЬ (analysis & persistence) ===
# ================================================================================


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

    table.add_row("Начальный капитал", f"\${metrics['Initial Capital']:.2f}")
    table.add_row("Финальный капитал", f"\${metrics['Final Equity']:.2f}")
    table.add_row(
        "Общий PnL", f"[{pnl_style}]\${metrics['Total PnL']:.2f}[/{pnl_style}]"
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


# ================================================================================
# === 5. ЛОГИКА ОПТИМИЗАЦИИ OPTUNA (С NUMBA) ===
# ================================================================================


def objective(
    trial: optuna.Trial, data_df: pd.DataFrame, base_config: StrategyConfig
) -> float:
    """
    Целевая функция для Optuna с Numba-ускорением.
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

    # 3. Расчет индикаторов и запуск бэктеста (с Numba)
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
    Основная функция для запуска оптимизации с помощью Optuna (с Numba).
    """

    metric_name = OptimizationParams.OPTIMIZATION_METRIC
    direction = "minimize" if metric_name == "Max Drawdown (%)" else "maximize"

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),  # Для воспроизводимости
    )

    console.print(
        f"\n[bold yellow]--- НАЧАЛО ОПТИМИЗАЦИИ OPTUNA (с Numba) ---[/bold yellow]"
    )
    console.print(
        f"Целевая метрика: [cyan]{metric_name}[/cyan] ([magenta]{direction}[/magenta])"
    )
    console.print(f"Количество попыток: {OptimizationParams.N_TRIALS}")

    study.optimize(
        lambda trial: objective(trial, data_df, base_config),
        n_trials=OptimizationParams.N_TRIALS,
        n_jobs=1,  # Numba лучше работает с n_jobs=1, параллелизм внутри функций
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


# ================================================================================
# === ОСНОВНАЯ ТОЧКА ВХОДА ===
# ================================================================================

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
                "\n[bold green]--- ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО (с Numba) ---[/bold green]"
            )
            console.print(f"Общее время выполнения: {execution_time_opt:.2f} сек.")

        else:
            console.print("\n[bold red]--- ОШИБКА ОПТИМИЗАЦИИ ---[/bold red]")
            console.print(
                "[bold red]Не удалось найти лучшую конфигурацию.[/bold red] Проверьте, совершила ли стратегия хотя бы одну сделку при заданных диапазонах параметров."
            )
