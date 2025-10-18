import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import matplotlib.pyplot as plt

# =========================================================================
# === МОДУЛЬ 6: АНАЛИЗ И ОТЧЕТНОСТЬ (RICH DASHBOARD) ===
# =========================================================================


def calculate_metrics(
    trades_df: pd.DataFrame, initial_capital: float, final_equity: float
):
    """Рассчитывает ключевые метрики бэктеста."""
    if trades_df.empty:
        # Если сделок нет, возвращаем пустые метрики
        return {}, pd.Series([0.0]), pd.Series([initial_capital])

    # Добавляем колонку 'liquidation' для совместимости, если ее нет (Mock)
    if "liquidation" not in trades_df.columns:
        trades_df["liquidation"] = 0

    total_pnl = trades_df["pnl"].sum()
    success_rate = (trades_df["pnl"] > 0).mean() * 100
    liquidation_rate = trades_df["liquidation"].mean() * 100
    num_trades = len(trades_df)

    return_on_capital = (final_equity - initial_capital) / initial_capital * 100

    # Расчет Максимальной Просадки (Max Drawdown - MDD)
    equity_curve = trades_df["equity_after_trade"]
    # Для расчета MDD нужно включить начальный капитал
    full_equity_curve = pd.concat([pd.Series([initial_capital]), equity_curve])
    peak = full_equity_curve.expanding().max()
    drawdown = (peak - full_equity_curve) / peak
    max_drawdown = drawdown.max() * 100 if not drawdown.empty else 0.0

    # MDD без включения стартового капитала (для графика)
    drawdown_for_plot = (
        equity_curve.expanding().max() - equity_curve
    ) / equity_curve.expanding().max()

    metrics = {
        "Начальный капитал": f"{initial_capital:,.2f}",
        "Финальный капитал": f"{final_equity:,.2f}",
        "Общий PNL": f"{total_pnl:,.2f}",
        "ROI (Прибыльность)": f"{return_on_capital:.2f}%",
        "Максимальная просадка (MDD)": f"{max_drawdown:.2f}%",
        "Общее количество сделок": f"{num_trades}",
        "Процент прибыльных сделок": f"{success_rate:.1f}%",
        "Процент ликвидаций": f"{liquidation_rate:.1f}%",
    }

    return metrics, drawdown_for_plot, equity_curve


def display_results_rich(
    metrics: dict[str, str], trades_df: pd.DataFrame, execution_time: float
):
    """Отображает результаты бэктеста в виде консольного дашборда (Rich)."""
    console = Console()

    # 1. Заголовок
    console.rule("[bold cyan]РЕЗУЛЬТАТЫ БЭКТЕСТА (Fibo Structure Scalper)[/bold cyan]")

    # 2. Таблица Основных Метрик
    table = Table(
        title="[bold white]СВОДКА МЕТРИК[/bold white]",
        show_lines=True,
        box=box.ROUNDED,
        style="dim",
    )
    table.add_column("Метрика", style="cyan", justify="left")
    table.add_column("Значение", style="yellow", justify="right")

    # Перевод ROI и PNL в float для проверки знака
    # Удаляем нечисловые символы
    pnl_str = metrics.get("Общий PNL", "0").replace("$", "").replace(",", "")
    try:
        total_pnl = float(pnl_str)
    except ValueError:
        total_pnl = 0.0

    for key, value in metrics.items():
        style = (
            "green"
            if ("ROI" in key or "PNL" in key) and total_pnl >= 0
            else (
                "red"
                if ("Просадка" in key or "Ликвидаций" in key or total_pnl < 0)
                else "white"
            )
        )
        table.add_row(key, f"[{style}]{value}[/{style}]")

    table.add_row("---", "---")
    table.add_row(
        "[bold]Время выполнения (Numba)[/bold]",
        f"[magenta]{execution_time:.4f} сек.[/magenta]",
    )

    console.print(
        Panel(table, title="[bold green]ДАШБОРД[/bold green]", border_style="green")
    )

    # 3. Лог Сделок (Первые 5 / Последние 5)
    if not trades_df.empty:
        log_cols = [
            "entry_time",
            "side",
            "entry_price",
            "exit_price",
            "pnl",
            "close_reason",
            "equity_after_trade",
        ]

        log_table = Table(
            title=f"[bold white]ЛОГ ЗАКРЫТЫХ СДЕЛОК (Показано 10 из {len(trades_df)})[/bold white]",
            box=box.MINIMAL,
        )
        for col in log_cols:
            log_table.add_column(
                col.replace("_", " ").capitalize(),
                style="dim",
                justify=(
                    "right"
                    if col not in ["close_reason", "side", "entry_time"]
                    else "left"
                ),
            )

        log_df = pd.concat([trades_df[log_cols].head(5), trades_df[log_cols].tail(5)])

        for _, row in log_df.iterrows():
            pnl_style = "green" if row["pnl"] >= 0 else "red"
            log_table.add_row(
                str(row["entry_time"])[:19],
                f"[bold blue]{row['side']}[/bold blue]",
                f"{row['entry_price']:,.2f}",
                f"{row['exit_price']:,.2f}",
                f"[{pnl_style}]{row['pnl']:,.2f}[/{pnl_style}]",
                f"[dim white]{row['close_reason']}[/dim white]",
                f"[bold]{row['equity_after_trade']:,.2f}[/bold]",
            )

        console.print(log_table)

    # 4. Графики (Примечание для пользователя)
    console.print(
        Panel(
            "[bold yellow]Графики (Equity Curve, Drawdown) будут отображены в отдельном окне Matplotlib.[/bold yellow]",
            border_style="yellow",
        )
    )


def plot_results(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    drawdown_for_plot: pd.Series,
    initial_capital: float,
):
    """Отображает графики эквити и просадки."""

    if equity_curve.empty:
        print("[PLOT] Недостаточно данных для построения графика.")
        return

    # Добавляем начальный капитал в начало кривой для корректного отображения
    equity_with_start = pd.concat([pd.Series([initial_capital]), equity_curve])

    # Создаем фиктивные индексы для графика, чтобы они соответствовали количеству сделок
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
    # Поскольку drawdown_for_plot рассчитывался на trades_df, нам нужно добавить 0.0 в начало
    drawdown_for_plot_with_start = (
        pd.concat([pd.Series([0.0]), drawdown_for_plot]) * 100
    )

    ax2.fill_between(
        trade_indices, 0, drawdown_for_plot_with_start.values, color="red", alpha=0.4
    )
    ax2.plot(
        trade_indices,
        drawdown_for_plot_with_start.values,
        color="red",
        alpha=0.8,
        label="Drawdown (%)",
    )
    ax2.set_title("Максимальная Текущая Просадка (Drawdown)", fontsize=14)
    ax2.set_xlabel("Номер Закрытой Сделки", fontsize=12)
    ax2.set_ylabel("Просадка (%)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}%"))

    plt.tight_layout()
    plt.show()
