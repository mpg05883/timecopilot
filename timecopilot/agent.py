from pathlib import Path
from typing import Callable

import fire
import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
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

from .models.benchmarks import (
    ADIDA,
    IMAPA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    DOTheta,
    HistoricAverage,
    SeasonalNaive,
    Theta,
    ZeroModel,
)
from .utils.experiment_handler import ExperimentDataset

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()

MODELS = {
    "ADIDA": ADIDA(),
    "AutoARIMA": AutoARIMA(),
    "AutoCES": AutoCES(),
    "AutoETS": AutoETS(),
    "CrostonClassic": CrostonClassic(),
    "DOTheta": DOTheta(),
    "HistoricAverage": HistoricAverage(),
    "IMAPA": IMAPA(),
    "SeasonalNaive": SeasonalNaive(),
    "Theta": Theta(),
    "ZeroModel": ZeroModel(),
}

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


class ForecastAgentOutput(BaseModel):
    features_used: list[str] = Field(
        description="Time series features that were considered"
    )
    selected_model: str = Field(
        description="The model that was selected for the forecast"
    )
    cross_validation_results: str = Field(description="The cross-validation results.")
    is_better_than_seasonal_naive: bool = Field(
        description="Whether the selected model is better than the seasonal naive model"
    )
    reason_for_selection: str = Field(
        description="Explanation for why the selected model was chosen"
    )
    forecast: list[float] = Field(
        description="The forecasted values for the time series"
    )
    user_prompt_response: str = Field(
        description="The response to the user's prompt, if any"
    )


forecasting_agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=ExperimentDataset,
    output_type=ForecastAgentOutput,
    system_prompt=f"""
    You're a forecasting expert. You will be given a time series as a list of numbers 
    and your task is to determine the best forecasting model for that series. You have 
    access to the following tools:

    1. tsfeatures_tool: Calculates time series features to help with model selection.
    Available features are: {", ".join(TSFEATURES.keys())}

    2. cross_validation_tool: Performs cross-validation for one or more models.
    Takes a list of model names and returns their cross-validation results.
    Available models are: {", ".join(MODELS.keys())}

    3. forecast_tool: Generates forecasts using a selected model.
    Takes a model name and returns forecasted values.

    Your task is to:
    1. Analyze the time series using tsfeatures_tool to understand its characteristics
    2. Based on the features, select promising models to evaluate
    3. Use cross_validation_tool to compare model performance
    4. Choose the best performing model that beats SeasonalNaive
    5. Generate final forecasts using forecast_tool
    6. If the user provides a prompt, use the generated forecast to generate a response

    The evaluation will use MASE (Mean Absolute Scaled Error) by default.
    Use at least one cross-validation window for evaluation.
    The seasonality will be inferred from the date column.

    For each step, explain your reasoning and decision-making process.
    Your final output must include:
    - Features analyzed and their implications
    - Models evaluated and their cross-validation results  
    - Rationale for the final model selection
    - Whether the chosen model beats SeasonalNaive
    - The forecasted values
    - The response to the user's prompt, if any
    """,
)


@forecasting_agent.system_prompt
async def add_time_series(ctx: RunContext[ExperimentDataset]) -> str:
    output = (
        f"The time series is: {ctx.deps.df['y'].tolist()}, "
        f"the date column is: {ctx.deps.df['ds'].tolist()}"
    )
    return output


@forecasting_agent.tool
async def tsfeatures_tool(
    ctx: RunContext[ExperimentDataset],
    features: list[str],
) -> str:
    features_df = tsfeatures(
        ctx.deps.df,
        features=[TSFEATURES[feature] for feature in features],
        freq=ctx.deps.seasonality,
    )
    return "\n".join(
        [f"{col}: {features_df[col].iloc[0]}" for col in features_df.columns]
    )


@forecasting_agent.tool
async def cross_validation_tool(
    ctx: RunContext[ExperimentDataset],
    models: list[str],
) -> str:
    models_fcst_cv = None
    callable_models = [MODELS[str_model] for str_model in models]
    for model in callable_models:
        fcst_cv = model.cross_validation(
            df=ctx.deps.df,
            h=ctx.deps.horizon,
            freq=ctx.deps.pandas_frequency,
        )
        if models_fcst_cv is None:
            models_fcst_cv = fcst_cv
        else:
            models_fcst_cv = models_fcst_cv.merge(
                fcst_cv.drop(columns=["y"]),
                on=["unique_id", "cutoff", "ds"],
            )
    eval_df = ctx.deps.evaluate_forecast_df(
        forecast_df=models_fcst_cv,
        models=[model.alias for model in callable_models],
    )
    eval_df = eval_df.groupby(["metric"], as_index=False).mean(numeric_only=True)
    return "\n".join(
        [f"{model.alias}: {eval_df[model.alias].iloc[0]}" for model in callable_models]
    )


@forecasting_agent.tool
async def forecast_tool(ctx: RunContext[ExperimentDataset], model: str) -> str:
    callable_model = MODELS[model]
    fcst_df = callable_model.forecast(
        df=ctx.deps.df,
        h=ctx.deps.horizon,
        freq=ctx.deps.pandas_frequency,
    )
    output = (
        f"Forecasted values for the next {ctx.deps.horizon} "
        f"periods: {fcst_df[model].tolist()}"
    )
    return output


@forecasting_agent.output_validator
async def validate_best_model(
    ctx: RunContext[ExperimentDataset],
    output: ForecastAgentOutput,
) -> ForecastAgentOutput:
    if not output.is_better_than_seasonal_naive:
        raise ModelRetry(
            "The selected model is not better than the seasonal naive model. "
            "Please try again with a different model."
            "The cross-validation results are: {output.cross_validation_results}"
        )
    return output


class TimeCopilot:
    async def forecast(self, path: str | Path, prompt: str = ""):
        dataset = ExperimentDataset.from_csv(path)
        result = await forecasting_agent.run(
            user_prompt=prompt,
            deps=dataset,
        )
        print(result.output)


if __name__ == "__main__":
    fire.Fire(TimeCopilot)
