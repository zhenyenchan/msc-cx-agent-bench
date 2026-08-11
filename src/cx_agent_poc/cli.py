"""Command-line interface for the agent.

Usage:
    cx-agent ask "What are the top complaints in Banking?"
    cx-agent ask "Summarise Fashion feedback" --verbose
    cx-agent benchmark
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cx_agent.data import load_fabsa
from cx_agent.tracing import run_with_trace
from cx_agent.evaluation import evaluate_record, run_benchmark, summarise

app = typer.Typer(help="CX analytics agent CLI")
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question for the agent"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full trace"),
) -> None:
    """Ask the agent a single question."""
    console.print(f"\n[bold cyan]Q:[/bold cyan] {question}\n")

    with console.status("Loading data..."):
        df_reviews, df_exploded = load_fabsa()

    with console.status("Thinking..."):
        record = run_with_trace(
            question=question,
            df_reviews=df_reviews,
            df_exploded=df_exploded,
        )

    # Show answer
    console.print(Panel(record["answer"], title="Answer", border_style="green"))

    # Show metrics
    metrics = evaluate_record(record)
    table = Table(title="Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)

    # Optional: show full trace
    if verbose:
        console.print("\n[bold]Full trace:[/bold]")
        for step in record["trace"]:
            console.print(step)


@app.command()
def benchmark() -> None:
    """Run the built-in 8-question benchmark."""
    test_cases = [
        {"question": "What are the top complaints in Banking?",
         "expected_tool": "describe", "question_type": "descriptive"},
        {"question": "What aspects are most mentioned in Fashion?",
         "expected_tool": "describe", "question_type": "descriptive"},
        {"question": "Is sentiment significantly different between app-website and speed?",
         "expected_tool": "infer", "question_type": "inferential"},
        {"question": "Compare sentiment between Banking and Fashion industries.",
         "expected_tool": "infer", "question_type": "inferential"},
        {"question": "Give me a summary report for the Fashion industry.",
         "expected_tool": "report", "question_type": "reporting"},
        {"question": "Summarise feedback from Google Play.",
         "expected_tool": "report", "question_type": "reporting"},
        {"question": "What is the capital of France?",
         "expected_tool": None, "question_type": "out_of_scope"},
        {"question": "What is the top trending handbag with best reviews?",
         "expected_tool": None, "question_type": "out_of_scope"},
    ]

    with console.status("Loading data..."):
        df_reviews, df_exploded = load_fabsa()

    console.print("\n[bold]Running benchmark...[/bold]\n")
    results = run_benchmark(test_cases, df_reviews, df_exploded)
    console.print()
    summarise(results)


if __name__ == "__main__":
    app()