from timecopilot.agent import MODELS

benchmark_models = ["AutoARIMA", "SeasonalNaive"]
models = [MODELS[str_model] for str_model in benchmark_models]
