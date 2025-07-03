import os

import pandas as pd
from dotenv import load_dotenv
from nixtla import NixtlaClient

from ..utils.forecaster import Forecaster

load_dotenv()


class TimeGPT(Forecaster):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 1,
        model: str = "timegpt-1",
        alias: str = "TimeGPT",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.model = model
        self.alias = alias

    def _get_client(self) -> NixtlaClient:
        if self.api_key is None:  # noqa: SIM108
            api_key = os.environ["NIXTLA_API_KEY"]
        else:
            api_key = self.api_key
        return NixtlaClient(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=self.max_retries,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        client = self._get_client()
        fcst_df = client.forecast(
            df=df,
            h=h,
            freq=freq,
            model=self.model,
            level=level,
            quantiles=quantiles,
        )
        fcst_df["ds"] = pd.to_datetime(fcst_df["ds"])
        fcst_df = fcst_df.rename(columns={"TimeGPT": self.alias})
        return fcst_df
