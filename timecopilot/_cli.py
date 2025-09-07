from pathlib import Path

import logfire
import typer
from dotenv import load_dotenv
from rich.console import Console

from timecopilot._interactive_cli import main as chat_main
from timecopilot.agent import TimeCopilot as TimeCopilotAgent

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


class TimeCopilot:
    def __init__(self):
        self.console = Console()

    def forecast(
        self,
        path: str | Path,
        llm: str = "openai:gpt-4o-mini",
        freq: str | None = None,
        h: int | None = None,
        seasonality: int | None = None,
        query: str | None = None,
        retries: int = 3,
    ):
        with self.console.status(
            "[bold blue]TimeCopilot is navigating through time...[/bold blue]"
        ):
            forecasting_agent = TimeCopilotAgent(llm=llm, retries=retries)
            result = forecasting_agent.forecast(
                df=path,
                freq=freq,
                h=h,
                seasonality=seasonality,
                query=query,
            )

        result.output.prettify(
            self.console,
            features_df=result.features_df,
            eval_df=result.eval_df,
            fcst_df=result.fcst_df,
        )


app = typer.Typer(
    name="timecopilot",
    help="TimeCopilot - Your GenAI Forecasting Agent",
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    llm: str = typer.Option(
        "openai:gpt-4o-mini", "--llm", "-l", help="LLM to use for the agent"
    ),
):
    """
    TimeCopilot - Your GenAI Forecasting Agent

    Start an interactive session by running 'timecopilot' with no arguments,
    or use specific commands for advanced usage.
    """
    if ctx.invoked_subcommand is None:
        import asyncio

        from timecopilot._interactive_cli import InteractiveTimeCopilot

        session = InteractiveTimeCopilot(llm=llm, use_async=True)
        asyncio.run(session.run_async())


@app.command()
def chat(
    llm: str = typer.Option(
        "openai:gpt-4o-mini", "--llm", "-l", help="LLM to use for the agent"
    ),
):
    """Start interactive TimeCopilot session (explicit command)."""
    chat_main()


@app.command("forecast")
def forecast_command(
    path: str = typer.Argument(..., help="Path to CSV file or URL"),
    llm: str = typer.Option(
        "openai:gpt-4o-mini", "--llm", "-l", help="LLM to use for forecasting"
    ),
    freq: str = typer.Option(None, "--freq", "-f", help="Data frequency"),
    h: int = typer.Option(None, "--horizon", "-h", help="Forecast horizon"),
    seasonality: int = typer.Option(None, "--seasonality", "-s", help="Seasonality"),
    query: str = typer.Option(None, "--query", "-q", help="Additional query"),
    retries: int = typer.Option(3, "--retries", "-r", help="Number of retries"),
):
    """Generate forecast (legacy one-shot mode)."""
    tc = TimeCopilot()
    tc.forecast(
        path=path,
        llm=llm,
        freq=freq,
        h=h,
        seasonality=seasonality,
        query=query,
        retries=retries,
    )


def main():
    app()


if __name__ == "__main__":
    main()
