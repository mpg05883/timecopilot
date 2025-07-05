from timecopilot.agent import MODELS

benchmark_models = [
    "AutoARIMA",
    "SeasonalNaive",
    "ZeroModel",
    "ADIDA",
    "TimesFM",
]
models = [MODELS[str_model] for str_model in benchmark_models]
