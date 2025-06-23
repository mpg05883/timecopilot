from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Union

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from tsfeatures import (
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    hurst,
    hw_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    series_length,
    stability,
    stl_features,
    tsfeatures,
    unitroot_kpss,
    unitroot_pp,
)

TSFEATURES: dict[str, Callable] = {
    "acf_features": acf_features,
    "arch_stat": arch_stat,
    "crossing_points": crossing_points,
    "entropy": entropy,
    "flat_spots": flat_spots,
    "heterogeneity": heterogeneity,
    "holt_parameters": holt_parameters,
    "lumpiness": lumpiness,
    "nonlinearity": nonlinearity,
    "pacf_features": pacf_features,
    "stl_features": stl_features,
    "stability": stability,
    "hw_parameters": hw_parameters,
    "unitroot_kpss": unitroot_kpss,
    "unitroot_pp": unitroot_pp,
    "series_length": series_length,
    "hurst": hurst,
}


@dataclass
class TimeSeries:
    y: List[float]
    ds: List[Union[str, datetime]]

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "TimeSeries":
        return cls(y=df["y"].tolist(), ds=df["ds"].tolist())

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({"ds": self.ds, "y": self.y})


@dataclass
class ForecastAgentOutput(BaseModel):
    forecast: List[float] = Field(
        description="The forecasted values for the time series"
    )
    selected_model: str = Field(
        description="The model that was selected for the forecast"
    )
    reason_for_selection: str = Field(
        description="Explanation for why the selected model was chosen"
    )
    features_used: List[str] = Field(
        description="Time series features that were considered"
    )


forecasting_agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=TimeSeries,
    output_type=ForecastAgentOutput,
    system_prompt=f"""
    You're a forecasting expert. You will be given a time series as a list of numbers 
    and your task is to determine the best forecasting model for that series. You have 
    access to the `tsfeatures_tool`, which allows you to calculate features of the time 
    series and, based on those, decide what model to try. The features that can receive 
    as input are:
    {", ".join(TSFEATURES.keys())}.

    You can select among these models as tools: ets, autoarima, theta, autotheta.

    Since time is limited, you need to be smart about model selection. Your target is 
    to beat a seasonal naive. You can compute the seasonal naive model using the 
    seasonal_naive tool.

    For evaluation, you have access to the cross_validation tool, which provides
    cross-validated forecasts, and the evaluate tool, which can be used to compare 
    model performance. If the user does not provide an evaluation metric, use MASE by 
    default. Infer the seasonality based on the date column if not explicitly provided. 
    Use at least one cross-validation window unless the user specifies otherwise.

    Document the whole process and explain your rationale to the user behind each 
    decision.
    """,
)


@forecasting_agent.system_prompt
async def add_time_series(ctx: RunContext[TimeSeries]) -> str:
    output = f"The time series is: {ctx.deps.y}, the date column is: {ctx.deps.ds}"
    return output


@forecasting_agent.tool
async def tsfeatures_tool(ctx: RunContext[TimeSeries], features: List[str]) -> str:
    df = ctx.deps.to_pandas()
    features_df = tsfeatures(
        df,
        features=[TSFEATURES[feature] for feature in features],
    )
    return "\n".join(
        [f"{col}: {features_df[col].iloc[0]}" for col in features_df.columns]
    )
