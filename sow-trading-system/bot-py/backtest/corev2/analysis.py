"""
ИСПРАВЛЕННЫЙ МОДУЛЬ АНАЛИЗА

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. Добавлен расчет Sharpe Ratio
2. Исправлена функция plot_results для работы с правильными параметрами
3. Улучшена обработка пустых данных
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import matplotlib.pyplot as plt


def calculate_metrics(
    trades_df: pd.DataFrame, initial_capital: float, final_equity: float
):
    """
    Рассчитывает ключевые метрики бэктеста, включая Sharpe Ratio.

    Returns:
        Tuple[Dict, pd.Series, pd.Series]: (metrics, drawdown, equity_curve)
    """
    if trades_df.empty:
        # Если сделок нет, возвращаем пустые метрики и кривую эквити
        return {}, pd.Series([0.0]), pd.Series([initial_capital])

    # Добавляем колонку 'liquidation' для совместимости, если ее нет
    if "liquidation" not in trades_df.columns:
        trades_df["liquidation"] = 0

    # Проверка наличия 'equity_after_trade' перед использованием
    if "equity_after_trade" not in trades_df.columns:
        # Если колонка отсутствует, рассчитываем ее
        trades_df["equity_after_trade"] = initial_capital + trades_df["pnl"].cumsum()

    equity_curve = trades_df["equity_after_trade"]

    # Расчет Максимальной Просадки (Max Drawdown - MDD)
    cumulative_max = equity_curve.cummax()
    drawdown = (cumulative_max - equity_curve) / cumulative_max

    # Основные метрики
    total_pnl = trades_df["pnl"].sum()
    success_rate = (trades_df["pnl"] > 0).mean() * 100
    liquidation_rate = trades_df["liquidation"].mean() * 100
    num_trades = len(trades_df)

    return_on_capital = (final_equity - initial_capital) / initial_capital * 100
    mdd = drawdown.max() * 100 if not drawdown.empty else 0.0

    # Расчет прибыли/убытка на сделку
    avg_pnl = trades_df["pnl"].mean()
    win_pnl = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    loss_pnl = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()

    # Расчет коэффициента прибыли (Profit Factor)
    total_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    total_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].abs().sum()
    profit_factor = total_win / total_loss if total_loss > 0 else np.inf

    # ===== ДОБАВЛЕНО: Расчет Sharpe Ratio =====
    # Sharpe Ratio = (Средняя доходность - Безрисковая ставка) / Стандартное отклонение доходности
    # Для простоты принимаем безрисковую ставку = 0

    if num_trades > 1:
        # Рассчитываем доходность на каждую сделку как процент от эквити до сделки
        # Используем pnl_perc из сделок
        if "pnl_perc" in trades_df.columns:
            returns = trades_df["pnl_perc"].values
        else:
            # Альтернативный расчет через изменение эквити
            equity_with_initial = np.concatenate(
                [[initial_capital], equity_curve.values[:-1]]
            )

            returns = (equity_curve.values - equity_with_initial) / equity_with_initial

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Используем несмещенную оценку

        if std_return > 0:
            # Аннуализируем Sharpe Ratio (предполагаем торговлю каждый день)
            # Для более точного расчета нужно знать частоту сделок
            sharpe_ratio = (
                mean_return / std_return * np.sqrt(252)
            )  # 252 торговых дня в году
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    metrics = {
        "Total Trades": num_trades,
        "Total PnL": total_pnl,
        "Final Equity": final_equity,
        "Return (%)": return_on_capital,
        "Success Rate (%)": success_rate,
        "Liquidation Rate (%)": liquidation_rate,
        "Max Drawdown (%)": mdd,
        "Avg PnL per Trade": avg_pnl,
        "Avg Win PnL": win_pnl,
        "Avg Loss PnL": loss_pnl,
        "Profit Factor": profit_factor,
        "Sharpe Ratio": sharpe_ratio,  # ДОБАВЛЕНО
    }

    # Возвращаем метрики, просадку и кривую эквити
    return metrics, drawdown, equity_curve


def display_results_rich(
    metrics: Dict[str, Any], trades_df: pd.DataFrame, execution_time: float
):
    """Отображение результатов с помощью библиотеки Rich."""
    console = Console()

    # Создание таблицы метрик
    table = Table(
        title="[bold green]Метрики Бэктеста[/bold green]",
        show_header=True,
        header_style="bold blue",
        box=box.MINIMAL_DOUBLE_HEAD,
    )

    # Определяем колонки и их стили
    table.add_column("Метрика", style="dim", justify="left")
    table.add_column("Значение", style="bold yellow", justify="right")

    # Добавление строк в таблицу
    def format_val(key, val):
        if isinstance(val, (int, np.integer)):
            return f"{val:,}"
        elif isinstance(val, (float, np.floating)):
            if key.endswith("(%)"):
                return f"{val:,.2f}%"
            elif key == "Profit Factor":
                return f"{val:,.2f}" if val != np.inf else "Inf"
            elif key == "Sharpe Ratio":
                return f"{val:,.2f}"
            else:
                return f"${val:,.2f}"
        else:
            return str(val)

    # Определяем порядок вывода метрик
    metric_order = [
        "Total Trades",
        "Final Equity",
        "Total PnL",
        "Return (%)",
        "Success Rate (%)",
        "Profit Factor",
        "Sharpe Ratio",  # ДОБАВЛЕНО
        "Max Drawdown (%)",
        "Avg PnL per Trade",
        "Avg Win PnL",
        "Avg Loss PnL",
        "Liquidation Rate (%)",
    ]

    for key in metric_order:
        if key in metrics:
            table.add_row(key, format_val(key, metrics[key]))

    # Панель для времени выполнения
    time_panel = Panel(
        f"[bold white]Время выполнения:[/bold white] [yellow]{execution_time:.2f} сек.[/yellow]\n"
        f"[bold white]Количество сделок:[/bold white] [cyan]{metrics.get('Total Trades', 0):,}[/cyan]",
        title="[bold magenta]Информация[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    )

    # Вывод
    console.print(table)
    console.print(time_panel)


def plot_results(
    trades_df: pd.DataFrame,
    initial_capital: float,
    equity_curve: pd.Series,
    drawdown_for_plot: pd.Series,
):
    """
    ИСПРАВЛЕННАЯ ВЕРСИЯ: Генерирует график кривой эквити и просадки.

    Args:
        trades_df: DataFrame со сделками
        initial_capital: Начальный капитал
        equity_curve: Series с кривой эквити после каждой сделки
        drawdown_for_plot: Series с просадкой после каждой сделки
    """

    # 1. Формируем полную кривую эквити, включая начальный капитал
    equity_with_start = pd.concat([pd.Series([initial_capital]), equity_curve])

    # Ось X должна соответствовать количеству точек данных
    trade_indices = np.arange(len(equity_with_start))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # График Эквити
    ax1.plot(
        trade_indices, equity_with_start.values, label="Equity Curve", color="green"
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
    drawdown_for_plot_with_start = (
        pd.concat([pd.Series([0.0]), drawdown_for_plot]) * 100
    )

    ax2.fill_between(
        trade_indices, 0, drawdown_for_plot_with_start.values, color="red", alpha=0.4
    )
    ax2.plot(
        trade_indices,
        drawdown_for_plot_with_start.values,
        label="Drawdown",
        color="red",
        linewidth=1,
    )
    ax2.set_title("Максимальная Просадка (Drawdown)", fontsize=14)
    ax2.set_xlabel("Номер Сделки", fontsize=12)
    ax2.set_ylabel("Просадка (%)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    plt.show()
