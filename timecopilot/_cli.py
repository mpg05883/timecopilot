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

    async def forecast(
        self,
        path: str | Path,
        model: str = "openai:gpt-4o-mini",
        prompt: str = "",
    ):
        with self.console.status("[bold green]Running forecast analysis..."):
            forecasting_agent = TimeCopilotAgent(model=model)
            result = await forecasting_agent.forecast(
                df=path,
                prompt=prompt,
            )

        result.output.prettify(self.console)
        return result


def main():
    fire.Fire(TimeCopilot)


if __name__ == "__main__":
    main()
