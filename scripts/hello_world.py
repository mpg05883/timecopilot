# Import libraries
import pandas as pd

from timecopilot import TimeCopilot

# Load the dataset
# The DataFrame must include at least the following columns:
# - unique_id: Unique identifier for each time series (string)
# - ds: Date column (datetime format)
# - y: Target variable for forecasting (float format)
# The pandas frequency will be inferred from the ds column, if not provided.
# If the seasonality is not provided, it will be inferred based on the
# frequency.
# If the horizon is not set, it will default to 2 times the inferred
# seasonality.
df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")

# Initialize the forecasting agent
# You can use any LLM by specifying the model parameter
tc = TimeCopilot(
    llm="openai:gpt-4o",
    retries=3,
)

# Generate forecast
# You can optionally specify the following parameters:
# - freq: The frequency of your data (e.g., 'D' for daily, 'M' for monthly)
# - h: The forecast horizon, which is the number of periods to predict
# - seasonality: The seasonal period of your data, which can be inferred if not
# provided
result = tc.forecast(df=df)

# The output contains:
# - tsfeatures_results: List of calculated time series features
# - tsfeatures_analysis: Natural language analysis of the features
# - selected_model: The best performing model chosen
# - model_details: Technical details about the selected model
# - cross_validation_results: Performance comparison of different models
# - model_comparison: Analysis of why certain models performed better/worse
# - is_better_than_seasonal_naive: Boolean indicating if model beats baseline
# - reason_for_selection: Explanation for model choice
# - forecast: List of future predictions with dates
# - forecast_analysis: Interpretation of the forecast results
# - user_query_response: Response to the user prompt, if any
print(result.output)
