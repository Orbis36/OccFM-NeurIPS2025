import numpy as np
from rich.table import Table
from rich.align import Align
from rich.panel import Panel
from rich.columns import Columns
from rich.box import SIMPLE_HEAD
from rich import box

from rich.console import Console

from rich.progress import (
    Progress, TextColumn, BarColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn
)

def format_disp_dict(disp_dict):
    table = Table(
        title="Current Loss",
        show_header=True,
        header_style="bold yellow",
        box=SIMPLE_HEAD,
        expand=False,
        pad_edge=False,
        padding=(0, 2),
    )

    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Value", justify="right", style="bold white")

    for key, val in disp_dict.items():
        if abs(val) < 1e-3 or abs(val) >= 1e4:
            value_str = f"{val:.5e}"
        else:
            value_str = f"{val:.4f}"

        table.add_row(str(key), value_str)

    panel = Panel(
        Align.center(table, vertical="middle"),
        border_style="dim",
        padding=(1, 4),
        width=60,  # 控制总宽度
    )

    return panel

def setup_loggers():

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[green]•[bold]{task.completed}[/] / {task.total}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

    console = Console(force_terminal=True, legacy_windows=True)

    return progress, console


def show_eval(avg_dict, console):
    all_miou, all_iou = avg_dict['all_miou'], avg_dict['all_iou']
    time_avg = avg_dict['time']
    mean_miou, mean_iou = np.mean(all_miou), np.mean(all_iou)

    if 'cate_miou' in avg_dict and isinstance(avg_dict['cate_miou'], dict):
        extra_table = Table(title="MIOU on each category", box=box.SIMPLE_HEAVY)
        for key in avg_dict['cate_miou'].keys():
            extra_table.add_column(str(key), justify="center", style="bold white")
        extra_table.add_row(*[f"{v:.2f}" if isinstance(v, float) else str(v) for v in avg_dict['cate_miou'].values()])
        console.print(extra_table)

    summary_panels = [
        Panel.fit(f"[bold green]{mean_miou:.4f}[/bold green]", title="Mean mIoU", border_style="magenta"),
        Panel.fit(f"[bold yellow]{mean_iou:.4f}[/bold yellow]", title="Mean IoU", border_style="cyan"),
        Panel.fit(f"[bold yellow]{time_avg:.4f}[/bold yellow]", title="Avg latency", border_style="orange1"),
    ]

    console.print(Columns(summary_panels))

    # 表格：Per-frame IoU 和 mIoU
    table = Table(title="Per-frame IoU and mIoU", box=box.ROUNDED, show_lines=True)

    table.add_column("Frame Index", justify="right", style="bold cyan")
    table.add_column("mIoU", justify="right", style="green")
    table.add_column("IoU", justify="right", style="yellow")

    num_frames = min(len(all_miou), len(all_iou))
    for idx in range(num_frames):
        table.add_row(str(idx), f"{all_miou[idx]:.4f}", f"{all_iou[idx]:.4f}")

    console.print(table)