# Time Series Model Hub


TimeCopilot provides a unified API for time series forecasting, integrating foundation models, classical statistical models, machine learning, and neural network families of models. This approach lets you experiment, benchmark, and deploy a wide range of forecasting models with minimal code changes, so you can choose the best tool for your data and use case.

Here you'll find all the time series forecasting models available in TimeCopilot, organized by family. Click on any model name to jump to its detailed API documentation.

!!! tip "Forecast multiple models using a unified API"

    With the [TimeCopilotForecaster][timecopilot.forecaster.TimeCopilotForecaster] class, you can generate and cross-validate forecasts using a unified API. Here's an example:

    ```python
    import pandas as pd
    from timecopilot.forecaster import TimeCopilotForecaster
    from timecopilot.models.benchmarks.prophet import Prophet
    from timecopilot.models.benchmarks.stats import AutoARIMA, SeasonalNaive
    from timecopilot.models.foundational.toto import Toto

    df = pd.read_csv(
        "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
        parse_dates=["ds"],
    )
    tcf = TimeCopilotForecaster(
        models=[
            AutoARIMA(),
            SeasonalNaive(),
            Prophet(),
            Toto(context_length=256),
        ]
    )

    fcst_df = tcf.forecast(df=df, h=12)
    cv_df = tcf.cross_validation(df=df, h=12)
    ```

---

## Foundation Models

TimeCopilot provides a unified interface to state-of-the-art foundation models for time series forecasting. These models are designed to handle a wide range of forecasting tasks, from classical seasonal patterns to complex, high-dimensional data. Below you will find a list of all available foundation models, each with a dedicated section describing its API and usage.

### [Chronos](api/models/foundational/models.md#timecopilot.models.foundational.chronos)
Large pre-trained transformer models for time series forecasting, supporting both probabilistic and point forecasts.

- **Paper**: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- **GitHub**: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- **HuggingFace**: [amazon/chronos-models](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444)

### [TimeGPT](api/models/foundational/models.md#timecopilot.models.foundational.timegpt)
A foundation model for time series forecasting from Nixtla, designed for production-ready forecasting with minimal setup.

- **Paper**: [TimeGPT-1](https://arxiv.org/abs/2310.03589)
- **GitHub**: [Nixtla/nixtla](https://github.com/Nixtla/nixtla)
- **HuggingFace**: [Nixtla Models](https://huggingface.co/Nixtla)

### [TimesFM](api/models/foundational/models.md#timecopilot.models.foundational.timesfm)
A decoder-only foundation model for time-series forecasting from Google Research, trained on diverse time series data.

- **Paper**: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)
- **GitHub**: [google-research/timesfm](https://github.com/google-research/timesfm)
- **HuggingFace**: [google/timesfm-release](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)

### [Toto](api/models/foundational/models.md#timecopilot.models.foundational.toto)
A foundation model for multivariate time series forecasting from Datadog, optimized for observability and high-dimensional data.

- **Paper**: [Building a Foundation Model for Time Series](https://arxiv.org/abs/2402.12971)
- **GitHub**: [DataDog/toto](https://github.com/DataDog/toto)
- **HuggingFace**: [Datadog Models](https://huggingface.co/Datadog)

### [TiRex](api/models/foundational/models.md#timecopilot.models.foundational.tirex)
A zero-shot time series forecasting model based on xLSTM, supporting both point and quantile predictions for long and short horizons.

- **Paper**: [TiRex: Zero-shot Time Series Forecasting with xLSTM](https://arxiv.org/abs/2412.11298)
- **GitHub**: [NX-AI/tirex](https://github.com/NX-AI/tirex)
- **HuggingFace**: [NX-AI Models](https://huggingface.co/NX-AI)

### [Moirai](api/models/foundational/models.md#timecopilot.models.foundational.moirai)
A universal foundation model for time series forecasting, designed to handle a wide range of frequencies, multivariate series, and covariates.

- **Paper**: [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)
- **GitHub**: [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
- **HuggingFace**: [Salesforce/moirai-r-models](https://huggingface.co/collections/Salesforce/moirai-r-models-65c8d3a94c51428c300e0742)

### [TabPFN](api/models/foundational/models.md#timecopilot.models.foundational.tabpfn)
A zero-shot time series forecasting model that frames univariate forecasting as a tabular regression problem using TabPFNv2.

- **Paper**: [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)
- **GitHub**: [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series)
- **HuggingFace**: [PriorLabs Models](https://huggingface.co/PriorLabs)

---

## Statistical & Classical Models

TimeCopilot includes a suite of classical and statistical forecasting models, providing robust baselines and interpretable alternatives to foundation models. These models are ideal for quick benchmarking, transparent forecasting, and scenarios where simplicity and speed are paramount. Below is a list of all available statistical models, each with a dedicated section describing its API and usage.

- [ADIDA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.ADIDA)
- [AutoARIMA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoARIMA)
- [AutoCES](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoCES)
- [AutoETS](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoETS)
- [CrostonClassic](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.CrostonClassic)
- [DynamicOptimizedTheta](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.DynamicOptimizedTheta)
- [HistoricAverage](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.HistoricAverage)
- [IMAPA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.IMAPA)
- [SeasonalNaive](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.SeasonalNaive)
- [Theta](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.Theta)
- [ZeroModel](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.ZeroModel)

---

## Prophet Model

TimeCopilot integrates the popular Prophet model for time series forecasting, developed by Facebook. Prophet is well-suited for business time series with strong seasonal effects and several seasons of historical data. Below you will find the API reference for the Prophet model.


- [Prophet](api/models/benchmarks/prophet.md/#timecopilot.models.benchmarks.prophet.Prophet)