from pathlib import Path

import fire
import logfire
from dotenv import load_dotenv
from rich.console import Console

from timecopilot.agent import TimeCopilot as TimeCopilotAgent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()


class TimeCopilot:
    def __init__(self):
        self.console = Console()

    def forecast(
        self,
        path: str | Path,
        model: str = "openai:gpt-4o-mini",
        prompt: str = "",
        retries: int = 3,
    ):
        with self.console.status(
            "[bold blue]TimeCopilot is navigating through time...[/bold blue]"
        ):
            forecasting_agent = TimeCopilotAgent(model=model, retries=retries)
            result = forecasting_agent.forecast(
                df=path,
                prompt=prompt,
            )

        result.output.prettify(self.console)


def main():
    fire.Fire(TimeCopilot)


if __name__ == "__main__":
    main()
