from pathlib import Path
from typing import Callable

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
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
    tsfeatures_results: list[str] = Field(
        description=(
            "The time series features that were considered as a list of strings of "
            "feature names and their values separated by commas."
        )
    )
    tsfeatures_analysis: str = Field(
        description=(
            "Analysis of what the time series features reveal about the data "
            "and their implications for forecasting."
        )
    )
    selected_model: str = Field(
        description="The model that was selected for the forecast"
    )
    model_details: str = Field(
        description=(
            "Technical details about the selected model including its assumptions, "
            "strengths, and typical use cases."
        )
    )
    cross_validation_results: list[str] = Field(
        description=(
            "The cross-validation results as a string of model names "
            "and their scores separated by commas."
        )
    )
    model_comparison: str = Field(
        description=(
            "Detailed comparison of model performances, explaining why certain "
            "models performed better or worse on this specific time series."
        )
    )
    is_better_than_seasonal_naive: bool = Field(
        description="Whether the selected model is better than the seasonal naive model"
    )
    reason_for_selection: str = Field(
        description="Explanation for why the selected model was chosen"
    )
    forecast: list[float] = Field(
        description="The forecasted values for the time series"
    )
    forecast_analysis: str = Field(
        description=(
            "Detailed interpretation of the forecast, including trends, patterns, "
            "and potential problems."
        )
    )
    user_prompt_response: str = Field(
        description="The response to the user's prompt, if any"
    )

    def prettify(self, console: Console | None = None) -> None:
        """Pretty print the forecast results using rich formatting."""
        console = console or Console()

        # Create main panel
        main_panel = Panel(
            "", title="[bold blue]Forecast Results[/bold blue]", expand=False
        )

        # Time Series Analysis Section
        features_analysis = Panel(
            f"[bold]Time Series Analysis:[/bold]\n{self.tsfeatures_analysis}",
            title="Feature Analysis",
            style="blue",
        )

        # Features table
        features_table = Table(title="Features Analyzed", show_header=True)
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Value", style="magenta")
        for feature in self.tsfeatures_results:
            feature_name, feature_value = feature.split(":")
            features_table.add_row(
                feature_name.strip(), f"{float(feature_value.strip()):.2f}"
            )

        # Model Selection Section
        model_info = Panel(
            f"[bold]Selected Model:[/bold] {self.selected_model}\n\n"
            f"[bold]Model Details:[/bold]\n{self.model_details}\n\n"
            f"[bold]Selection Rationale:[/bold]\n{self.reason_for_selection}\n\n"
            "[bold]Performance vs Seasonal Naive:[/bold] "
            f"{'✓' if self.is_better_than_seasonal_naive else '✗'}",
            title="Model Selection",
            style="green",
        )

        # Cross validation results
        cv_table = Table(title="Cross Validation Results", show_header=True)
        cv_table.add_column("Model", style="cyan")
        cv_table.add_column("Score", style="magenta")
        for result_line in self.cross_validation_results:
            model, score = result_line.split(":")
            cv_table.add_row(model.strip(), f"{float(score.strip()):.2f}")

        model_comparison_panel = Panel(
            f"[bold]Model Comparison:[/bold]\n{self.model_comparison}",
            title="Performance Analysis",
            style="yellow",
        )

        # Forecast Section
        forecast_table = Table(title="Forecast Values", show_header=True)
        forecast_table.add_column("Period", style="cyan")
        forecast_table.add_column("Value", style="magenta")
        for i, value in enumerate(self.forecast, 1):
            forecast_table.add_row(f"t+{i}", f"{value:.2f}")

        forecast_details = Panel(
            f"[bold]Forecast Analysis:[/bold]\n{self.forecast_analysis}",
            title="Forecast Details",
            style="magenta",
        )

        # User prompt response if exists
        prompt_panel = None
        if self.user_prompt_response:
            prompt_panel = Panel(
                self.user_prompt_response,
                title="Response to User Prompt",
                style="italic",
            )

        # Print everything with clear sections
        console.print("\n")
        console.print(main_panel)

        # Time Series Analysis
        console.print("[bold]1. Time Series Analysis[/bold]")
        console.print(features_table)
        console.print(features_analysis)

        # Model Selection and Evaluation
        console.print("\n[bold]2. Model Selection and Evaluation[/bold]")
        console.print(model_info)
        console.print(cv_table)
        console.print(model_comparison_panel)

        # Forecast Results
        console.print("\n[bold]3. Forecast Results[/bold]")
        console.print(forecast_table)
        console.print(forecast_details)

        if prompt_panel:
            console.print("\n[bold]4. User Response[/bold]")
            console.print(prompt_panel)

        console.print("\n")


class TimeCopilot:
    def __init__(
        self,
        **kwargs,
    ):
        self.system_prompt = f"""
        You're a forecasting expert. You will be given a time series 
        as a list of numbers
        and your task is to determine the best forecasting model for that series. 
        You have access to the following tools:

        1. tsfeatures_tool: Calculates time series features to help 
        with model selection.
        Available features are: {", ".join(TSFEATURES.keys())}

        2. cross_validation_tool: Performs cross-validation for one or more models.
        Takes a list of model names and returns their cross-validation results.
        Available models are: {", ".join(MODELS.keys())}

        3. forecast_tool: Generates forecasts using a selected model.
        Takes a model name and returns forecasted values.

        Your task is to provide a comprehensive analysis following these steps:

        1. Time Series Feature Analysis:
           - Calculate relevant time series features
           - Analyze what these features reveal about the data's nature
           - Explain the implications for forecasting strategy
           - Identify key characteristics that will influence model selection

        2. Model Selection and Evaluation:
           - Select candidate models based on the time series values and features
           - Document each model's technical details and assumptions
           - Explain why these models are suitable for the identified features
           - Perform cross-validation to evaluate performance
           - Compare models quantitatively and qualitatively
           - Verify performance against the seasonal naive benchmark

        3. Final Model Selection and Forecasting:
           - Choose the best performing model with clear justification
           - Generate and analyze the forecast
           - Interpret trends and patterns in the forecast
           - Discuss reliability and potential uncertainties
           - Address any specific aspects from the user's prompt

        The evaluation will use MASE (Mean Absolute Scaled Error) by default.
        Use at least one cross-validation window for evaluation.
        The seasonality will be inferred from the date column.

        Your output must include:
        - Comprehensive feature analysis with clear implications
        - Detailed model comparison and selection rationale
        - Technical details of the selected model
        - Clear interpretation of cross-validation results
        - Analysis of the forecast and its implications
        - Response to any user queries

        Focus on providing:
        - Clear connections between features and model choices
        - Technical accuracy with accessible explanations
        - Quantitative support for decisions
        - Practical implications of the forecast
        - Thorough responses to user concerns
        """
        self.forecasting_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=ForecastAgentOutput,
            system_prompt=self.system_prompt,
            **kwargs,
        )

        @self.forecasting_agent.system_prompt
        async def add_time_series(ctx: RunContext[ExperimentDataset]) -> str:
            output = (
                f"The time series is: {ctx.deps.df['y'].tolist()}, "
                f"the date column is: {ctx.deps.df['ds'].tolist()}"
            )
            return output

        @self.forecasting_agent.tool
        async def tsfeatures_tool(
            ctx: RunContext[ExperimentDataset],
            features: list[str],
        ) -> str:
            callable_features = []
            for feature in features:
                if feature not in TSFEATURES:
                    raise ModelRetry(
                        f"Feature {feature} is not available. Available features are: "
                        f"{', '.join(TSFEATURES.keys())}"
                    )
                callable_features.append(TSFEATURES[feature])
            features_df = tsfeatures(
                ctx.deps.df,
                features=callable_features,
                freq=ctx.deps.seasonality,
            )
            features_df = features_df.drop(columns=["unique_id"])
            return ",".join(
                [f"{col}: {features_df[col].iloc[0]}" for col in features_df.columns]
            )

        @self.forecasting_agent.tool
        async def cross_validation_tool(
            ctx: RunContext[ExperimentDataset],
            models: list[str],
        ) -> str:
            models_fcst_cv = None
            callable_models = []
            for str_model in models:
                if str_model not in MODELS:
                    raise ModelRetry(
                        f"Model {str_model} is not available. Available models are: "
                        f"{', '.join(MODELS.keys())}"
                    )
                callable_models.append(MODELS[str_model])
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
            eval_df = eval_df.groupby(
                ["metric"],
                as_index=False,
            ).mean(numeric_only=True)
            return ", ".join(
                [
                    f"{model.alias}: {eval_df[model.alias].iloc[0]}"
                    for model in callable_models
                ]
            )

        @self.forecasting_agent.tool
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

        @self.forecasting_agent.output_validator
        async def validate_best_model(
            ctx: RunContext[ExperimentDataset],
            output: ForecastAgentOutput,
        ) -> ForecastAgentOutput:
            if not output.is_better_than_seasonal_naive:
                raise ModelRetry(
                    "The selected model is not better than the seasonal naive model. "
                    "Please try again with a different model."
                    "The cross-validation results are: "
                    "{output.cross_validation_results}"
                )
            return output

    async def forecast(self, df: pd.DataFrame | str | Path, prompt: str = ""):
        if isinstance(df, (str, Path)):
            dataset = ExperimentDataset.from_csv(df)
        elif isinstance(df, pd.DataFrame):
            dataset = ExperimentDataset.from_df(df=df)
        else:
            raise ValueError(f"Invalid input type: {type(df)}")

        result = await self.forecasting_agent.run(
            user_prompt=prompt,
            deps=dataset,
        )

        return result
