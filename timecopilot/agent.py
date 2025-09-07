from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.agent import AgentRunResult
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
    unitroot_kpss,
    unitroot_pp,
)
from tsfeatures.tsfeatures import _get_feats

from .forecaster import Forecaster, TimeCopilotForecaster
from .models.prophet import Prophet
from .models.stats import (
    ADIDA,
    IMAPA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    DynamicOptimizedTheta,
    HistoricAverage,
    SeasonalNaive,
    Theta,
    ZeroModel,
)
from .utils.experiment_handler import ExperimentDataset, ExperimentDatasetParser

DEFAULT_MODELS: list[Forecaster] = [
    ADIDA(),
    AutoARIMA(),
    AutoCES(),
    AutoETS(),
    CrostonClassic(),
    DynamicOptimizedTheta(),
    HistoricAverage(),
    IMAPA(),
    SeasonalNaive(),
    Theta(),
    ZeroModel(),
    Prophet(),
]

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


class TimeCopilotAgentOutput(BaseModel):
    """The output of the TimeCopilot agent for any type of analysis."""

    analysis_type: str = Field(
        description=(
            "The type of analysis performed: 'forecasting', 'anomaly_detection', "
            "'visualization', or 'combined'"
        )
    )

    # Data understanding (common to all workflows)
    data_analysis: str = Field(
        description=(
            "Analysis of the time series data characteristics, features, and patterns "
            "relevant to the requested analysis type."
        )
    )

    # Forecasting-specific fields (optional)
    selected_model: str | None = Field(
        default=None,
        description="The model that was selected (for forecasting workflows)",
    )
    model_details: str | None = Field(
        default=None,
        description=(
            "Technical details about the selected model including its assumptions, "
            "strengths, and typical use cases."
        ),
    )
    model_comparison: str | None = Field(
        default=None,
        description=(
            "Detailed comparison of model performances, explaining why certain "
            "models performed better or worse on this specific time series."
        ),
    )
    is_better_than_seasonal_naive: bool | None = Field(
        default=None,
        description=(
            "Whether the selected model is better than the seasonal naive model"
        ),
    )
    forecast_analysis: str | None = Field(
        default=None,
        description=(
            "Detailed interpretation of the forecast, including trends, patterns, "
            "and potential problems."
        ),
    )

    # Anomaly detection fields (optional)
    anomaly_analysis: str | None = Field(
        default=None,
        description=(
            "Analysis of detected anomalies, their patterns, potential causes, "
            "and recommendations for handling them."
        ),
    )

    # Visualization fields (optional)
    visualization_description: str | None = Field(
        default=None,
        description=(
            "Description of generated visualizations and key insights visible "
            "in the charts."
        ),
    )

    # Common fields
    main_findings: str = Field(
        description=(
            "The main findings and insights from the analysis, tailored to "
            "the specific type of analysis requested."
        )
    )
    recommendations: str = Field(
        description=("Actionable recommendations based on the analysis results.")
    )
    user_query_response: str | None = Field(
        default=None,
        description=(
            "The response to the user's query, if any. "
            "If the user did not provide a query, this field will be None."
        ),
    )

    def prettify(
        self,
        console: Console | None = None,
        features_df: pd.DataFrame | None = None,
        eval_df: pd.DataFrame | None = None,
        fcst_df: pd.DataFrame | None = None,
        anomalies_df: pd.DataFrame | None = None,
    ) -> None:
        """Pretty print the analysis results using rich formatting."""
        console = console or Console()

        # Create header based on analysis type
        if self.analysis_type == "forecasting":
            title = "TimeCopilot Forecast Analysis"
            model_info = (
                f"[bold cyan]{self.selected_model}[/bold cyan] forecast analysis"
            )
            if self.is_better_than_seasonal_naive is not None:
                model_info += (
                    f"\n[{'green' if self.is_better_than_seasonal_naive else 'red'}]"
                )
                is_better = self.is_better_than_seasonal_naive
                better_text = "✓ Better" if is_better else "✗ Not better"
                color = "green" if self.is_better_than_seasonal_naive else "red"
                model_info += f"{better_text} than Seasonal Naive[/{color}]"
        elif self.analysis_type == "anomaly_detection":
            title = "TimeCopilot Anomaly Detection"
            model_info = "[bold red]🚨 Anomaly Detection Analysis[/bold red]"
        elif self.analysis_type == "visualization":
            title = "TimeCopilot Visualization"
            model_info = "[bold green]📊 Data Visualization[/bold green]"
        else:  # combined
            title = "TimeCopilot Analysis"
            model_info = "[bold purple]🔍 Combined Analysis[/bold purple]"

        header = Panel(
            model_info,
            title=f"[bold blue]{title}[/bold blue]",
            style="blue",
        )

        # Time Series Analysis Section - check if features_df is available
        ts_features = Table(
            title="Time Series Features",
            show_header=True,
            title_style="bold cyan",
            header_style="bold magenta",
        )
        ts_features.add_column("Feature", style="cyan")
        ts_features.add_column("Value", style="magenta")

        # Use features_df if available (attached after forecast run)
        if features_df is not None:
            for feature_name, feature_value in features_df.iloc[0].items():
                if pd.notna(feature_value):
                    ts_features.add_row(feature_name, f"{float(feature_value):.3f}")
        else:
            # Fallback: show a note that detailed features are not available
            ts_features.add_row("Features", "Available in analysis text below")

        ts_analysis = Panel(
            f"{self.data_analysis}",
            title="[bold cyan]Data Analysis[/bold cyan]",
            style="blue",
        )

        # Analysis-specific sections
        analysis_sections = []

        # Forecasting sections
        if self.analysis_type in ["forecasting", "combined"] and self.model_details:
            model_details = Panel(
                f"[bold]Technical Details[/bold]\n{self.model_details}\n\n"
                f"[bold]Model Comparison[/bold]\n{self.model_comparison or 'N/A'}",
                title="[bold green]Model Information[/bold green]",
                style="green",
            )
            analysis_sections.append(("Model Selection", model_details))

        # Anomaly detection sections
        if (
            self.analysis_type in ["anomaly_detection", "combined"]
            and self.anomaly_analysis
        ):
            anomaly_details = Panel(
                self.anomaly_analysis,
                title="[bold red]Anomaly Analysis[/bold red]",
                style="red",
            )
            analysis_sections.append(("Anomaly Detection", anomaly_details))

        # Visualization sections
        if (
            self.analysis_type in ["visualization", "combined"]
            and self.visualization_description
        ):
            viz_details = Panel(
                self.visualization_description,
                title="[bold green]Visualization Details[/bold green]",
                style="green",
            )
            analysis_sections.append(("Visualization", viz_details))

        # Model Comparison Table - check if eval_df is available
        model_scores = Table(
            title="Model Performance", show_header=True, title_style="bold yellow"
        )
        model_scores.add_column("Model", style="yellow")
        model_scores.add_column("MASE", style="cyan", justify="right")

        # Use eval_df if available (attached after forecast run)
        if eval_df is not None:
            # Get the MASE scores from eval_df
            model_scores_data = []
            for col in eval_df.columns:
                if col != "metric" and pd.notna(eval_df[col].iloc[0]):
                    model_scores_data.append((col, float(eval_df[col].iloc[0])))

            # Sort by score (lower MASE is better)
            model_scores_data.sort(key=lambda x: x[1])
            for model, score in model_scores_data:
                model_scores.add_row(model, f"{score:.3f}")
        else:
            # Fallback: show a note that detailed scores are not available
            model_scores.add_row("Scores", "Available in analysis text below")

        model_analysis = Panel(
            self.model_comparison,
            title="[bold yellow]Performance Analysis[/bold yellow]",
            style="yellow",
        )

        # Forecast Results Section - check if fcst_df is available
        forecast_table = Table(
            title="Forecast Values", show_header=True, title_style="bold magenta"
        )
        forecast_table.add_column("Period", style="magenta")
        forecast_table.add_column("Value", style="cyan", justify="right")

        # Use fcst_df if available (attached after forecast run)
        if fcst_df is not None:
            # Show forecast values from fcst_df
            fcst_data = fcst_df.copy()
            if "ds" in fcst_data.columns and self.selected_model in fcst_data.columns:
                for _, row in fcst_data.iterrows():
                    period = (
                        row["ds"].strftime("%Y-%m-%d")
                        if hasattr(row["ds"], "strftime")
                        else str(row["ds"])
                    )
                    value = row[self.selected_model]
                    forecast_table.add_row(period, f"{value:.2f}")

                # Add note about number of periods if many
                if len(fcst_data) > 12:
                    forecast_table.caption = (
                        f"[dim]Showing all {len(fcst_data)} forecasted periods. "
                        "Use aggregation functions for summarized views.[/dim]"
                    )
            else:
                forecast_table.add_row("Forecast", "Available in analysis text below")
        else:
            # Fallback: show a note that detailed forecast is not available
            forecast_table.add_row("Forecast", "Available in analysis text below")

        # Main findings and recommendations
        main_findings = Panel(
            f"[bold]Key Findings[/bold]\n{self.main_findings}\n\n"
            f"[bold]Recommendations[/bold]\n{self.recommendations}",
            title="[bold magenta]Analysis Results[/bold magenta]",
            style="magenta",
        )

        # Optional user response section
        user_response = None
        if self.user_query_response:
            user_response = Panel(
                self.user_query_response,
                title="[bold]Response to Query[/bold]",
                style="cyan",
            )

        # Print all sections with clear separation
        console.print("\n")
        console.print(header)

        console.print("\n[bold]1. Data Analysis[/bold]")
        console.print(ts_features)
        console.print(ts_analysis)

        # Print analysis-specific sections
        section_num = 2
        for section_name, section_panel in analysis_sections:
            console.print(f"\n[bold]{section_num}. {section_name}[/bold]")
            console.print(section_panel)
            section_num += 1

        # Show performance table only for forecasting
        if self.analysis_type in ["forecasting", "combined"] and eval_df is not None:
            console.print(model_scores)
            console.print(model_analysis)

        # Show forecast table only for forecasting
        if self.analysis_type in ["forecasting", "combined"] and fcst_df is not None:
            console.print(f"\n[bold]{section_num}. Results[/bold]")
            console.print(forecast_table)
            section_num += 1

        # Always show main findings
        console.print(f"\n[bold]{section_num}. Summary[/bold]")
        console.print(main_findings)

        if user_response:
            console.print(f"\n[bold]{section_num + 1}. Additional Information[/bold]")
            console.print(user_response)

        console.print("\n")


def _transform_time_series_to_text(df: pd.DataFrame) -> str:
    df_agg = df.groupby("unique_id").agg(list)
    output = (
        "these are the time series in json format where the key is the "
        "identifier of the time series and the values is also a json "
        "of two elements: "
        "the first element is the date column and the second element is the "
        "value column."
        f"{df_agg.to_json(orient='index')}"
    )
    return output


def _transform_features_to_text(features_df: pd.DataFrame) -> str:
    output = (
        "these are the time series features in json format where the key is "
        "the identifier of the time series and the values is also a json of "
        "feature names and their values."
        f"{features_df.to_json(orient='index')}"
    )
    return output


def _transform_eval_to_text(eval_df: pd.DataFrame, models: list[str]) -> str:
    output = ", ".join([f"{model}: {eval_df[model].iloc[0]}" for model in models])
    return output


def _transform_fcst_to_text(fcst_df: pd.DataFrame) -> str:
    df_agg = fcst_df.groupby("unique_id").agg(list)
    output = (
        "these are the forecasted values in json format where the key is the "
        "identifier of the time series and the values is also a json of two "
        "elements: the first element is the date column and the second "
        "element is the value column."
        f"{df_agg.to_json(orient='index')}"
    )
    return output


def _transform_anomalies_to_text(anomalies_df: pd.DataFrame) -> str:
    """Transform anomaly detection results to text for the agent."""
    # Get anomaly columns
    anomaly_cols = [col for col in anomalies_df.columns if col.endswith("-anomaly")]

    if not anomaly_cols:
        return "No anomaly detection results available."

    # Count anomalies per series
    anomaly_summary = {}
    for unique_id in anomalies_df["unique_id"].unique():
        series_data = anomalies_df[anomalies_df["unique_id"] == unique_id]
        series_summary = {}

        for anomaly_col in anomaly_cols:
            if anomaly_col in series_data.columns:
                anomaly_count = series_data[anomaly_col].sum()
                total_points = len(series_data)
                anomaly_rate = (
                    (anomaly_count / total_points) * 100 if total_points > 0 else 0
                )

                # Get timestamps of anomalies
                anomalies = series_data[series_data[anomaly_col]]
                anomaly_dates = (
                    anomalies["ds"].dt.strftime("%Y-%m-%d").tolist()
                    if len(anomalies) > 0
                    else []
                )

                series_summary[anomaly_col] = {
                    "count": int(anomaly_count),
                    "rate_percent": round(anomaly_rate, 2),
                    "dates": anomaly_dates[:10],  # Limit to first 10
                    "total_points": int(total_points),
                }

        anomaly_summary[unique_id] = series_summary

    output = (
        "these are the anomaly detection results in json format where the key is the "
        "identifier of the time series and the values contain anomaly statistics "
        "including count, rate, and timestamps of detected anomalies. "
        f"{anomaly_summary}"
    )
    return output


class TimeCopilot:
    """
    TimeCopilot: An AI agent for comprehensive time series analysis.

    Supports multiple analysis workflows:
    - Forecasting: Predict future values
    - Anomaly Detection: Identify outliers and unusual patterns
    - Visualization: Generate plots and charts
    - Combined: Multiple analysis types together
    """

    # Tool organization by workflow
    FORECASTING_TOOLS = ["tsfeatures_tool", "cross_validation_tool", "forecast_tool"]
    ANOMALY_TOOLS = ["tsfeatures_tool", "detect_anomalies_tool"]
    VISUALIZATION_TOOLS = ["plot_tool"]
    SHARED_TOOLS = ["tsfeatures_tool"]  # Used across multiple workflows

    def __init__(
        self,
        llm: str,
        forecasters: list[Forecaster] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM to use.
            forecasters: A list of forecasters to use. If not provided,
                TimeCopilot will use the default forecasters.
            **kwargs: Additional keyword arguments to pass to the agent.
        """

        if forecasters is None:
            forecasters = DEFAULT_MODELS
        self.forecasters = {forecaster.alias: forecaster for forecaster in forecasters}
        if "SeasonalNaive" not in self.forecasters:
            self.forecasters["SeasonalNaive"] = SeasonalNaive()
        self.system_prompt = f"""
        You are TimeCopilot, a time series analysis expert. You will be given time 
        series data and your task is to provide comprehensive analysis based on the 
        user's request.

        ## TASK IDENTIFICATION:
        First, carefully analyze the user query to identify the primary task type:

         FORECASTING KEYWORDS: "forecast", "predict", "future", "projection", 
           "ahead", "next", "coming"
         ANOMALY DETECTION KEYWORDS: "anomaly", "anomalies", "outlier", "outliers", 
           "unusual", "abnormal", "irregular", "detect", "find"
         VISUALIZATION KEYWORDS: "plot", "chart", "graph", "visualize", "show", 
           "display", "draw"
         COMBINED KEYWORDS: Multiple types mentioned (e.g., "forecast and detect 
           anomalies")

        ## AVAILABLE TOOLS:
        You have access to these specialized tools organized by workflow:

        ### FORECASTING TOOLS:
        1. tsfeatures_tool: Calculate time series characteristics for model selection
           Available features: {", ".join(TSFEATURES.keys())}
        2. cross_validation_tool: Evaluate model performance across time windows
           Available models: {", ".join(self.forecasters.keys())}
        3. forecast_tool: Generate predictions using selected model

        ### ANOMALY DETECTION TOOLS:
        1. tsfeatures_tool: Understand data patterns for anomaly context
        2. detect_anomalies_tool: Identify outliers using statistical methods

        ### VISUALIZATION TOOLS:
        1. plot_tool: Create charts and graphs (types: 'forecast', 'anomalies', 'both')

        ## WORKFLOW EXECUTION:
        Based on your task identification, execute the appropriate workflow:

        ### A) FORECASTING WORKFLOW
        **Trigger**: User wants predictions, future values, or forecasts
        **Tools**: tsfeatures_tool → cross_validation_tool → forecast_tool
        **Steps**:
        1. **Data Understanding** (REQUIRED - tsfeatures_tool):
           ✓ Call tsfeatures_tool with key features: ["stl_features", "acf_features", 
             "seasonality", "trend"]
           ✓ Identify seasonality, trend, stationarity, and complexity patterns
           ✓ Use insights to guide model selection strategy

        2. **Model Evaluation** (REQUIRED - cross_validation_tool):
           ✓ Test multiple models based on data characteristics
           ✓ Start with simple models (SeasonalNaive, AutoETS, AutoARIMA)
           ✓ Add complex models if simple ones fail to beat SeasonalNaive
           ✓ Compare performance using MASE metric
           ✓ Document why each model is suitable for the data

        3. **Final Forecasting** (REQUIRED - forecast_tool):
           ✓ Select best-performing model with clear justification
           ✓ Generate forecast with selected model only
           ✓ Interpret trends, seasonality, and forecast reliability
           ✓ Address user-specific questions about the forecast

        ### B) ANOMALY DETECTION WORKFLOW  
        **Trigger**: User wants to find outliers, anomalies, or unusual patterns
        **Tools**: tsfeatures_tool → detect_anomalies_tool
        **Steps**:
        1. **Pattern Analysis** (RECOMMENDED - tsfeatures_tool):
           ✓ Focus on variability features: ["stability", "lumpiness", "entropy"]
           ✓ Understand normal patterns to identify what's anomalous
           ✓ Consider seasonality for context-aware anomaly detection

        2. **Anomaly Detection** (REQUIRED - detect_anomalies_tool):
           ✓ Choose appropriate model (SeasonalNaive for seasonal data, AutoARIMA 
             for complex patterns)
           ✓ Set confidence level (95% typical, 99% for stricter detection)
           ✓ Analyze detected anomalies and their timing
           ✓ Explain statistical basis for anomaly identification

        3. **Anomaly Interpretation**:
           ✓ Describe patterns in detected anomalies
           ✓ Discuss potential causes (seasonal effects, external events)
           ✓ Provide actionable recommendations for handling anomalies

        ### C) VISUALIZATION WORKFLOW
        **Trigger**: User wants plots, charts, graphs, or visual analysis
        **Tools**: plot_tool (+ data generation tools if needed)
        **Steps**:
        1. **Visualization Planning**:
           ✓ Determine what data needs to be visualized
           ✓ Choose appropriate plot type based on request
           ✓ Identify if additional analysis is needed first

        2. **Data Generation** (if needed):
           ✓ Use forecast_tool if forecast visualization is requested
           ✓ Use detect_anomalies_tool if anomaly visualization is requested
           ✓ Ensure all required data is available for plotting

        3. **Plot Creation** (REQUIRED - plot_tool):
           ✓ Generate appropriate visualizations with plot_tool
           ✓ Include relevant models and highlight key insights
           ✓ Explain what the visualizations reveal about the data

        ### D) COMBINED WORKFLOWS
        **Trigger**: Multiple analysis types requested (e.g., "forecast and detect 
        anomalies")
        **Strategy**: Execute multiple workflows and integrate results
        **Steps**:
        1. **Workflow Identification**: Identify all requested analysis types
        2. **Sequential Execution**: Run each workflow with shared context
        3. **Result Integration**: Combine insights from all analyses
        4. **Unified Visualization**: Use plot_tool with 'both' type if appropriate

        The evaluation will use MASE (Mean Absolute Scaled Error) by default.
        Use at least one cross-validation window for evaluation.
        The seasonality will be inferred from the date column.

        ## OUTPUT REQUIREMENTS:
        Your response must be structured based on the identified workflow:

        ### FOR FORECASTING WORKFLOW:
        ✓ **Data Analysis**: Time series characteristics and feature insights
        ✓ **Model Comparison**: Quantitative performance comparison with rationale
        ✓ **Selected Model**: Technical details and why it was chosen
        ✓ **Forecast Results**: Clear interpretation of predictions and trends
        ✓ **Reliability Assessment**: Confidence intervals and uncertainty discussion

        ### FOR ANOMALY DETECTION WORKFLOW:
        ✓ **Pattern Context**: Normal data characteristics for anomaly context
        ✓ **Detection Results**: Number, timing, and severity of anomalies
        ✓ **Statistical Basis**: Confidence levels and detection methodology
        ✓ **Anomaly Analysis**: What makes these points unusual
        ✓ **Actionable Insights**: Recommendations for handling detected anomalies

        ### FOR VISUALIZATION WORKFLOW:
        ✓ **Plot Description**: What visualizations were generated
        ✓ **Visual Insights**: Key patterns and trends visible in charts
        ✓ **Data Story**: What the visualizations reveal about the time series
        ✓ **Interpretation Guide**: How to read and understand the plots

        ### FOR COMBINED WORKFLOWS:
        ✓ **Integrated Analysis**: How different analyses complement each other
        ✓ **Cross-Workflow Insights**: Connections between forecasts, anomalies, 
          and visualizations
        ✓ **Unified Recommendations**: Comprehensive advice based on all analyses

        ### UNIVERSAL REQUIREMENTS (ALL WORKFLOWS):
        ✓ **User Query Response**: Direct answer to specific user questions
        ✓ **Technical Accuracy**: Correct methodology with accessible explanations
        ✓ **Quantitative Support**: Numbers and metrics backing up conclusions
        ✓ **Actionable Recommendations**: Practical next steps for the user

        ## CRITICAL INSTRUCTIONS:
        🎯 **Task Matching**: Execute ONLY the workflow that matches the user's request
        🚫 **No Workflow Forcing**: Don't run forecasting if user only wants 
           anomaly detection
        🔄 **Context Preservation**: Use shared insights across tools within the 
           same workflow
        📊 **Analysis Type Setting**: Always set the correct analysis_type in 
           your output
        """

        if "model" in kwargs:
            raise ValueError(
                "model is not allowed to be passed as a keyword argument"
                "use `llm` instead"
            )
        self.llm = llm

        self.forecasting_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=TimeCopilotAgentOutput,
            system_prompt=self.system_prompt,
            model=self.llm,
            **kwargs,
        )

        self.query_system_prompt = """
        You are a TimeCopilot assistant with conversation memory. You have access to 
        dataframes from previous analysis and can create visualizations.

        AVAILABLE DATAFRAMES (may vary based on analysis type):
        - fcst_df: Forecasted values for each time series, including dates and 
          predicted values.
        - eval_df: Evaluation results for each model. The evaluation metric is always 
          MASE (Mean Absolute Scaled Error), as established in the main system prompt. 
          Each value in eval_df represents the MASE score for a model.
        - features_df: Extracted time series features for each series, such as trend, 
          seasonality, autocorrelation, and more.
        - anomalies_df: Anomaly detection results (if available), including timestamps,
          actual values, predictions, and anomaly flags.

        CONVERSATION CONTEXT:
        You maintain conversation history and can understand references to previous 
        exchanges. When users say "plot them", "show me", "visualize it", etc., 
        use context from the conversation to understand what they're referring to.

        VISUALIZATION CAPABILITIES:
        You CAN create plots and visualizations! When users ask for plots, charts, 
        or visualizations, explain that you can generate them and describe what 
        type of visualization would be most appropriate based on the available data.

        RESPONSE GUIDELINES:
        - Use conversation history to understand context and references
        - Reference specific values, trends, or metrics from the dataframes
        - For plotting requests, confirm you can create visualizations
        - If data is missing for a request, explain what's available
        - Always explain your reasoning and cite relevant data

        You can help users understand:
        - Future trends and predictions (from fcst_df)
        - Model reliability and confidence (from eval_df)
        - Seasonal patterns and cycles (from features_df)
        - Anomalous behavior and outliers (from anomalies_df)
        - Comparative model performance
        - Data quality and characteristics
        - Create appropriate visualizations for any of the above

        IMPORTANT: You have full visualization capabilities. Never say you cannot 
        create plots - instead, describe what visualization would be helpful and 
        confirm you can generate it.
        """

        self.query_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=str,
            system_prompt=self.query_system_prompt,
            model=self.llm,
            **kwargs,
        )

        self.dataset: ExperimentDataset
        self.fcst_df: pd.DataFrame
        self.eval_df: pd.DataFrame
        self.features_df: pd.DataFrame
        self.anomalies_df: pd.DataFrame
        self.eval_forecasters: list[str]

        # Conversation history for maintaining context between queries
        self.conversation_history: list[dict] = []

        @self.query_agent.system_prompt
        async def add_experiment_info(
            ctx: RunContext[ExperimentDataset],
        ) -> str:
            output_parts = [
                _transform_time_series_to_text(ctx.deps.df),
            ]

            # Add dataframes if they exist (depends on workflow)
            if hasattr(self, "features_df") and self.features_df is not None:
                output_parts.append(_transform_features_to_text(self.features_df))

            if (
                hasattr(self, "eval_df")
                and self.eval_df is not None
                and hasattr(self, "eval_forecasters")
                and self.eval_forecasters is not None
            ):
                output_parts.append(
                    _transform_eval_to_text(self.eval_df, self.eval_forecasters)
                )

            if hasattr(self, "fcst_df") and self.fcst_df is not None:
                output_parts.append(_transform_fcst_to_text(self.fcst_df))

            if hasattr(self, "anomalies_df") and self.anomalies_df is not None:
                anomaly_text = _transform_anomalies_to_text(self.anomalies_df)
                output_parts.append(anomaly_text)

            return "\n".join(output_parts)

        @self.forecasting_agent.system_prompt
        async def add_time_series(
            ctx: RunContext[ExperimentDataset],
        ) -> str:
            return _transform_time_series_to_text(ctx.deps.df)

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
            features_dfs = []
            for uid in ctx.deps.df["unique_id"].unique():
                features_df_uid = _get_feats(
                    index=uid,
                    ts=ctx.deps.df,
                    features=callable_features,
                    freq=ctx.deps.seasonality,
                )
                features_dfs.append(features_df_uid)
            features_df = pd.concat(features_dfs) if features_dfs else pd.DataFrame()
            features_df = features_df.rename_axis("unique_id")  # type: ignore
            self.features_df = features_df
            return _transform_features_to_text(features_df)

        @self.forecasting_agent.tool
        async def cross_validation_tool(
            ctx: RunContext[ExperimentDataset],
            models: list[str],
        ) -> str:
            callable_models = []
            for str_model in models:
                if str_model not in self.forecasters:
                    raise ModelRetry(
                        f"Model {str_model} is not available. Available models are: "
                        f"{', '.join(self.forecasters.keys())}"
                    )
                callable_models.append(self.forecasters[str_model])
            forecaster = TimeCopilotForecaster(models=callable_models)
            fcst_cv = forecaster.cross_validation(
                df=ctx.deps.df,
                h=ctx.deps.h,
                freq=ctx.deps.freq,
            )
            eval_df = ctx.deps.evaluate_forecast_df(
                forecast_df=fcst_cv,
                models=[model.alias for model in callable_models],
            )
            eval_df = eval_df.groupby(
                ["metric"],
                as_index=False,
            ).mean(numeric_only=True)
            self.eval_df = eval_df
            self.eval_forecasters = models
            return _transform_eval_to_text(eval_df, models)

        @self.forecasting_agent.tool
        async def forecast_tool(
            ctx: RunContext[ExperimentDataset],
            model: str,
        ) -> str:
            callable_model = self.forecasters[model]
            forecaster = TimeCopilotForecaster(models=[callable_model])
            fcst_df = forecaster.forecast(
                df=ctx.deps.df,
                h=ctx.deps.h,
                freq=ctx.deps.freq,
            )
            self.fcst_df = fcst_df
            return _transform_fcst_to_text(fcst_df)

        @self.forecasting_agent.tool
        async def detect_anomalies_tool(
            ctx: RunContext[ExperimentDataset],
            model: str,
            level: int = 95,
        ) -> str:
            """
            Detect anomalies in the time series using the specified model.

            Args:
                model: The model to use for anomaly detection
                level: Confidence level for anomaly detection (default: 95)
            """
            callable_model = self.forecasters[model]
            anomalies_df = callable_model.detect_anomalies(
                df=ctx.deps.df,
                freq=ctx.deps.freq,
                level=level,
            )
            self.anomalies_df = anomalies_df

            # Transform to text for the agent
            anomaly_count = anomalies_df[f"{model}-anomaly"].sum()
            total_points = len(anomalies_df)
            anomaly_rate = (
                (anomaly_count / total_points) * 100 if total_points > 0 else 0
            )

            output = (
                f"Anomaly detection completed using {model} model. "
                f"Found {anomaly_count} anomalies out of {total_points} data points "
                f"({anomaly_rate:.1f}% anomaly rate) at {level}% confidence level. "
                f"Anomalies are flagged in the '{model}-anomaly' column."
            )

            if anomaly_count > 0:
                # Add details about detected anomalies
                anomalies = anomalies_df[anomalies_df[f"{model}-anomaly"]]
                timestamps = list(anomalies["ds"].dt.strftime("%Y-%m-%d").head(10))
                output += f" Anomalies detected at timestamps: {timestamps}"
                if len(anomalies) > 10:
                    output += f" and {len(anomalies) - 10} more."

            return output

        @self.forecasting_agent.tool
        async def plot_tool(
            ctx: RunContext[ExperimentDataset],
            plot_type: str = "forecast",
            models: list[str] | None = None,
        ) -> str:
            """
            Generate plots for the time series data and results.

            Args:
                plot_type: Type of plot ('forecast', 'anomalies', 'both')
                models: List of models to include in the plot
            """
            try:
                from timecopilot.models.utils.forecaster import Forecaster

                # Determine what to plot based on available data
                if plot_type == "forecast" and hasattr(self, "fcst_df"):
                    # Plot forecast results
                    if models is None:
                        # Use all available models in forecast
                        model_cols = [
                            col
                            for col in self.fcst_df.columns
                            if col not in ["unique_id", "ds"] and "-" not in col
                        ]
                        models = model_cols

                    Forecaster.plot(
                        df=ctx.deps.df,
                        forecasts_df=self.fcst_df,
                        models=models,
                        max_ids=5,
                        engine="matplotlib",
                    )
                    return f"Generated forecast plot for models: {', '.join(models)}"

                elif plot_type == "anomalies" and hasattr(self, "anomalies_df"):
                    # Plot anomaly detection results
                    Forecaster.plot(
                        df=None,  # Will be inferred from anomalies_df
                        forecasts_df=self.anomalies_df,
                        plot_anomalies=True,
                        max_ids=5,
                        engine="matplotlib",
                    )
                    return "Generated anomaly detection plot"

                elif plot_type == "both":
                    # Plot both if available
                    plots_generated = []

                    if hasattr(self, "fcst_df"):
                        Forecaster.plot(
                            df=ctx.deps.df,
                            forecasts_df=self.fcst_df,
                            max_ids=5,
                            engine="matplotlib",
                        )
                        plots_generated.append("forecast")

                    if hasattr(self, "anomalies_df"):
                        Forecaster.plot(
                            df=None,
                            forecasts_df=self.anomalies_df,
                            plot_anomalies=True,
                            max_ids=5,
                            engine="matplotlib",
                        )
                        plots_generated.append("anomalies")

                    if plots_generated:
                        return f"Generated plots for: {', '.join(plots_generated)}"
                    else:
                        return (
                            "No data available for plotting. Please run forecast "
                            "or anomaly detection first."
                        )

                else:
                    return (
                        f"Cannot generate {plot_type} plot. Required data not "
                        "available. Please run the appropriate analysis first."
                    )

            except Exception as e:
                return f"Error generating plot: {str(e)}"

        @self.forecasting_agent.output_validator
        async def validate_analysis_output(
            ctx: RunContext[ExperimentDataset],
            output: TimeCopilotAgentOutput,
        ) -> TimeCopilotAgentOutput:
            """Workflow-specific validation based on analysis type."""

            if output.analysis_type == "forecasting":
                return self._validate_forecasting_output(output)
            elif output.analysis_type == "anomaly_detection":
                return self._validate_anomaly_output(output)
            elif output.analysis_type == "visualization":
                return self._validate_visualization_output(output)
            elif output.analysis_type == "combined":
                return self._validate_combined_output(output)
            else:
                valid_types = (
                    "'forecasting', 'anomaly_detection', 'visualization', 'combined'"
                )
                raise ModelRetry(
                    f"Unknown analysis_type: {output.analysis_type}. "
                    f"Must be one of: {valid_types}"
                )

    def _validate_forecasting_output(
        self, output: TimeCopilotAgentOutput
    ) -> TimeCopilotAgentOutput:
        """Validate forecasting workflow output."""
        # Check required forecasting fields
        if not output.selected_model:
            raise ModelRetry("Forecasting workflow must specify a selected_model.")

        if not output.model_details:
            raise ModelRetry("Forecasting workflow must provide model_details.")

        if not output.forecast_analysis:
            raise ModelRetry("Forecasting workflow must provide forecast_analysis.")

        # Check model performance requirement
        if (
            output.is_better_than_seasonal_naive is not None
            and not output.is_better_than_seasonal_naive
        ):
            raise ModelRetry(
                "The selected model is not better than the seasonal naive model. "
                "Please try again with a different model. "
                f"Cross-validation results: {output.model_comparison}"
            )

        return output

    def _validate_anomaly_output(
        self, output: TimeCopilotAgentOutput
    ) -> TimeCopilotAgentOutput:
        """Validate anomaly detection workflow output."""
        if not output.anomaly_analysis:
            raise ModelRetry(
                "Anomaly detection workflow must provide anomaly_analysis."
            )

        # Check that anomaly-specific insights are provided
        if "anomal" not in output.main_findings.lower():
            raise ModelRetry(
                "Anomaly detection workflow must include anomaly insights in "
                "main_findings."
            )

        return output

    def _validate_visualization_output(
        self, output: TimeCopilotAgentOutput
    ) -> TimeCopilotAgentOutput:
        """Validate visualization workflow output."""
        if not output.visualization_description:
            raise ModelRetry(
                "Visualization workflow must provide visualization_description."
            )

        # Check that visualization insights are provided
        if not any(
            word in output.main_findings.lower()
            for word in ["plot", "chart", "graph", "visual"]
        ):
            raise ModelRetry(
                "Visualization workflow must include visualization insights in "
                "main_findings."
            )

        return output

    def _validate_combined_output(
        self, output: TimeCopilotAgentOutput
    ) -> TimeCopilotAgentOutput:
        """Validate combined workflow output."""
        # Count how many workflow types are actually present
        workflow_count = 0

        if output.selected_model and output.forecast_analysis:
            workflow_count += 1

        if output.anomaly_analysis:
            workflow_count += 1

        if output.visualization_description:
            workflow_count += 1

        if workflow_count < 2:
            raise ModelRetry(
                "Combined workflow must include results from at least 2 different "
                f"analysis types. Currently only {workflow_count} workflow type(s) "
                "detected."
            )

        return output

    def is_queryable(self) -> bool:
        """
        Check if the class is queryable.
        It needs to have `dataset` and at least one analysis result dataframe.
        """
        # Must have dataset
        if not (hasattr(self, "dataset") and self.dataset is not None):
            return False

        # Must have at least one result dataframe from any workflow
        has_forecasting = hasattr(self, "fcst_df") and self.fcst_df is not None
        has_anomalies = hasattr(self, "anomalies_df") and self.anomalies_df is not None
        has_features = hasattr(self, "features_df") and self.features_df is not None

        return has_forecasting or has_anomalies or has_features

    def analyze(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[TimeCopilotAgentOutput]:
        """Analyze time series data with forecasting, anomaly detection, or
        visualization.

        This method can handle multiple types of analysis based on the query:
        - Forecasting: Generate predictions for future periods
        - Anomaly Detection: Identify outliers and unusual patterns
        - Visualization: Create plots and charts
        - Combined: Multiple analysis types together

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments. Examples:
                - "forecast next 12 months"
                - "detect anomalies with 95% confidence"
                - "plot the time series data"
                - "forecast and detect anomalies"

        Returns:
            A result object whose `output` attribute is a fully
                populated [`TimeCopilotAgentOutput`][timecopilot.agent.
                TimeCopilotAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        query = f"User query: {query}" if query else None
        experiment_dataset_parser = ExperimentDatasetParser(
            model=self.forecasting_agent.model,
        )
        self.dataset = experiment_dataset_parser.parse(
            df,
            freq,
            h,
            seasonality,
            query,
        )
        result = self.forecasting_agent.run_sync(
            user_prompt=query,
            deps=self.dataset,
        )
        # Attach dataframes if they exist (depends on workflow)
        result.fcst_df = getattr(self, "fcst_df", None)
        result.eval_df = getattr(self, "eval_df", None)
        result.features_df = getattr(self, "features_df", None)
        result.anomalies_df = getattr(self, "anomalies_df", None)
        return result

    def forecast(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[TimeCopilotAgentOutput]:
        """Generate forecast and analysis.

        .. deprecated:: 0.1.0
            Use :meth:`analyze` instead. This method is kept for backwards
            compatibility.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`TimeCopilotAgentOutput`][timecopilot.agent.
                TimeCopilotAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        # Delegate to the new analyze method
        return self.analyze(df=df, h=h, freq=freq, seasonality=seasonality, query=query)

    def _maybe_raise_if_not_queryable(self):
        if not self.is_queryable():
            raise ValueError(
                "The class is not queryable. Please run analysis first using "
                "`analyze()` or `forecast()`."
            )

    def query(
        self,
        query: str,
    ) -> AgentRunResult[str]:
        # fmt: off
        """
        Ask a follow-up question about the analysis results with conversation history.

        This method enables chat-like, interactive querying after an analysis
        has been run. The agent will use the stored dataframes and maintain
        conversation history to provide contextual responses. It can answer
        questions about forecasts, anomalies, visualizations, and more.

        Args:
            query: The user's follow-up question. This can be about model
                performance, forecast results, anomaly detection, or visualizations.

        Returns:
            AgentRunResult[str]: The agent's answer as a string. Use
                `result.output` to access the answer.

        Raises:
            ValueError: If the class is not ready for querying (i.e., no analysis
                has been run and required dataframes are missing).

        Example:
            ```python
            import pandas as pd
            from timecopilot import TimeCopilot

            df = pd.read_csv("data.csv") 
            tc = TimeCopilot(llm="openai:gpt-4o")
            
            # Run anomaly detection
            tc.analyze(df, query="detect anomalies")
            
            # Follow-up with conversation history
            answer = tc.query("plot them")  # "them" refers to the anomalies
            print(answer.output)
            ```
        Note:
            The class is not queryable until an analysis method has been called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()

        # Build conversation context with history
        conversation_context = self._build_conversation_context(query)

        result = self.query_agent.run_sync(
            user_prompt=conversation_context,
            deps=self.dataset,
        )

        # Store the conversation in history
        self.conversation_history.append({"user": query, "assistant": result.output})

        return result

    def _build_conversation_context(self, current_query: str) -> str:
        """Build conversation context including history for better responses."""
        if not self.conversation_history:
            # No history, just return the current query
            return current_query

        # Build context with conversation history
        context_parts = ["Previous conversation:"]

        # Add recent conversation history (last 5 exchanges to avoid token limits)
        recent_history = self.conversation_history[-5:]
        for exchange in recent_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")

        context_parts.append(f"\nCurrent question: {current_query}")

        return "\n".join(context_parts)

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []


class AsyncTimeCopilot(TimeCopilot):
    def __init__(self, **kwargs: Any):
        """
        Initialize an asynchronous TimeCopilot agent.

        Inherits from TimeCopilot and provides async methods for
        forecasting and querying.
        """
        super().__init__(**kwargs)

    async def analyze(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[TimeCopilotAgentOutput]:
        """
        Asynchronously analyze time series data with forecasting, anomaly detection,
        or visualization.

        This method can handle multiple types of analysis based on the query:
        - Forecasting: Generate predictions for future periods
        - Anomaly Detection: Identify outliers and unusual patterns
        - Visualization: Create plots and charts
        - Combined: Multiple analysis types together

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - You must always work with time series data with the columns
                  ds (date) and y (target value), if these are missing, attempt to
                  infer them from similar column names or, if unsure, request
                  clarification from the user.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments. Examples:
                - "forecast next 12 months"
                - "detect anomalies with 95% confidence"
                - "plot the time series data"
                - "forecast and detect anomalies"

        Returns:
            A result object whose `output` attribute is a fully
                populated [`TimeCopilotAgentOutput`][timecopilot.agent.
                TimeCopilotAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        query = f"User query: {query}" if query else None
        experiment_dataset_parser = ExperimentDatasetParser(
            model=self.forecasting_agent.model,
        )
        self.dataset = await experiment_dataset_parser.parse_async(
            df,
            freq,
            h,
            seasonality,
            query,
        )
        result = await self.forecasting_agent.run(
            user_prompt=query,
            deps=self.dataset,
        )
        # Attach dataframes if they exist (depends on workflow)
        result.fcst_df = getattr(self, "fcst_df", None)
        result.eval_df = getattr(self, "eval_df", None)
        result.features_df = getattr(self, "features_df", None)
        result.anomalies_df = getattr(self, "anomalies_df", None)
        return result

    async def forecast(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[TimeCopilotAgentOutput]:
        """
        Asynchronously generate forecast and analysis for the provided
        time series data.

        .. deprecated:: 0.1.0
            Use :meth:`analyze` instead. This method is kept for backwards
            compatibility.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - You must always work with time series data with the columns
                  ds (date) and y (target value), if these are missing, attempt to
                  infer them from similar column names or, if unsure, request
                  clarification from the user.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`TimeCopilotAgentOutput`][timecopilot.agent.
                TimeCopilotAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        # Delegate to the new analyze method
        return await self.analyze(
            df=df, h=h, freq=freq, seasonality=seasonality, query=query
        )

    @asynccontextmanager
    async def query_stream(
        self,
        query: str,
    ) -> AsyncGenerator[AgentRunResult[str], None]:
        # fmt: off
        """
        Asynchronously stream the agent's answer to a follow-up question.

        This method enables chat-like, interactive querying after a forecast 
        has been run.
        The agent will use the stored dataframes and the original dataset 
        to answer the user's
        question, yielding results as they become available (streaming).

        Args:
            query: The user's follow-up question. This can be about model
                performance, forecast results, or time series features.

        Returns:
            AgentRunResult[str]: The agent's answer as a string. Use
                `result.output` to access the answer.

        Raises:
            ValueError: If the class is not ready for querying (i.e., forecast
                has not been run and required dataframes are missing).

        Example:
            ```python
            import asyncio

            import pandas as pd
            from timecopilot import AsyncTimeCopilot

            df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv") 

            async def example():
                tc = AsyncTimeCopilot(llm="openai:gpt-4o")
                await tc.forecast(df, h=12, freq="MS")
                async with tc.query_stream("Which model performed best?") as result:
                    async for text in result.stream(debounce_by=0.01):
                        print(text, end="", flush=True)
            
            asyncio.run(example())
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()

        # Build conversation context with history
        conversation_context = self._build_conversation_context(query)

        async with self.query_agent.run_stream(
            user_prompt=conversation_context,
            deps=self.dataset,
        ) as result:
            # Store the conversation in history after streaming completes
            # Note: We'll store the final result when the stream is consumed
            yield result

            # Store conversation after streaming (this might not capture the full
            # response)
            # For streaming, we'll store what we can
            self.conversation_history.append(
                {"user": query, "assistant": "[Streaming response - see above]"}
            )

    async def query(
        self,
        query: str,
    ) -> AgentRunResult[str]:
        # fmt: off
        """
        Asynchronously ask a follow-up question about the forecast, 
        model evaluation, or time series features.

        This method enables chat-like, interactive querying after a forecast
        has been run. The agent will use the stored dataframes (`fcst_df`,
        `eval_df`, `features_df`) and the original dataset to answer the user's
        question in a data-driven manner. Typical queries include asking about
        the best model, forecasted values, or time series characteristics.

        Args:
            query: The user's follow-up question. This can be about model
                performance, forecast results, or time series features.

        Returns:
            AgentRunResult[str]: The agent's answer as a string. Use
                `result.output` to access the answer.

        Raises:
            ValueError: If the class is not ready for querying (i.e., forecast
                has not been run and required dataframes are missing).

        Example:
            ```python
            import asyncio

            import pandas as pd
            from timecopilot import AsyncTimeCopilot

            df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv") 

            async def example():
                tc = AsyncTimeCopilot(llm="openai:gpt-4o")
                await tc.forecast(df, h=12, freq="MS")
                answer = await tc.query("Which model performed best?")
                print(answer.output)

            asyncio.run(example())
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()

        # Build conversation context with history
        conversation_context = self._build_conversation_context(query)

        result = await self.query_agent.run(
            user_prompt=conversation_context,
            deps=self.dataset,
        )

        # Store the conversation in history
        self.conversation_history.append({"user": query, "assistant": result.output})

        return result
