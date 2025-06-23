from typing import Callable, List

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
    features_used: List[str] = Field(
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
    forecast: List[float] = Field(
        description="The forecasted values for the time series"
    )


forecasting_agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=ExperimentDataset,
    output_type=ForecastAgentOutput,
    system_prompt=f"""
    You're a forecasting expert. You will be given a time series as a list of numbers 
    and your task is to determine the best forecasting model for that series. You have 
    access to the `tsfeatures_tool`, which allows you to calculate features of the time 
    series and, based on those, decide what model to try. The features that can receive 
    as input are:
    {", ".join(TSFEATURES.keys())}.

    You can select among these models as tools: {", ".join(MODELS.keys())}.

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
async def add_time_series(ctx: RunContext[ExperimentDataset]) -> str:
    output = (
        f"The time series is: {ctx.deps.df['y'].tolist()}, "
        f"the date column is: {ctx.deps.df['ds'].tolist()}"
    )
    return output


@forecasting_agent.tool
async def tsfeatures_tool(
    ctx: RunContext[ExperimentDataset],
    features: List[str],
) -> str:
    features_df = tsfeatures(
        ctx.deps.df,
        features=[TSFEATURES[feature] for feature in features],
    )
    return "\n".join(
        [f"{col}: {features_df[col].iloc[0]}" for col in features_df.columns]
    )


@forecasting_agent.tool
async def cross_validation_tool(
    ctx: RunContext[ExperimentDataset],
    models: List[str],
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
                fcst_cv.drop(columns=["y"]), on=["unique_id", "ds"]
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
