# Time Series Model Hub


!!! abstract "Welcome to the Time Series Model Hub!"
    TimeCopilot provides a unified API for time series forecasting, integrating foundation models, classical statistical models, machine learning, and neural network families of models. This approach lets you experiment, benchmark, and deploy a wide range of forecasting models with minimal code changes, so you can choose the best tool for your data and use case.

    Here you'll find all the time series forecasting models available in TimeCopilot, organized by family. Click on any model name to jump to its detailed API documentation.

---

## Foundation Models

TimeCopilot provides a unified interface to state-of-the-art foundation models for time series forecasting. These models are designed to handle a wide range of forecasting tasks, from classical seasonal patterns to complex, high-dimensional data. Below you will find a list of all available foundation models, each with a dedicated section describing its API and usage.

- [Chronos](api/models/foundational/models.md#timecopilot.models.foundational.chronos)
- [TimeGPT](api/models/foundational/models.md#timecopilot.models.foundational.timegpt)
- [TimesFM](api/models/foundational/models.md#timecopilot.models.foundational.timesfm)
- [Toto](api/models/foundational/models.md#timecopilot.models.foundational.toto)
- [TiRex](api/models/foundational/models.md#timecopilot.models.foundational.tirex)
- [LagLlama](api/models/foundational/models.md#timecopilot.models.foundational.lagllama)
- [Moirai](api/models/foundational/models.md#timecopilot.models.foundational.moirai)

---

## Statistical & Classical Models

TimeCopilot includes a suite of classical and statistical forecasting models, providing robust baselines and interpretable alternatives to foundation models. These models are ideal for quick benchmarking, transparent forecasting, and scenarios where simplicity and speed are paramount. Below is a list of all available statistical models, each with a dedicated section describing its API and usage.

- [ADIDA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.ADIDA)
- [AutoARIMA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoARIMA)
- [AutoCES](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoCES)
- [AutoETS](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoETS)
- [CrostonClassic](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.CrostonClassic)
- [DOTheta](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.DOTheta)
- [HistoricAverage](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.HistoricAverage)
- [IMAPA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.IMAPA)
- [SeasonalNaive](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.SeasonalNaive)
- [Theta](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.Theta)
- [ZeroModel](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.ZeroModel)

---

## Prophet Model

TimeCopilot integrates the popular Prophet model for time series forecasting, developed by Facebook. Prophet is well-suited for business time series with strong seasonal effects and several seasons of historical data. Below you will find the API reference for the Prophet model.


- [Prophet](api/models/benchmarks/prophet.md/#timecopilot.models.benchmarks.prophet.Prophet)