from pathlib import Path

import fire
import logfire
from dotenv import load_dotenv

from timecopilot.agent import TimeCopilot as TimeCopilotAgent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()


class TimeCopilot:
    async def forecast(
        self,
        path: str | Path,
        model: str = "openai:gpt-4o-mini",
        prompt: str = "",
    ):
        forecasting_agent = TimeCopilotAgent(model=model)
        result = await forecasting_agent.forecast(
            df=path,
            prompt=prompt,
        )
        print(result.output)


def main():
    fire.Fire(TimeCopilot)


if __name__ == "__main__":
    main()
