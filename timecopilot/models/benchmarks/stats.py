import os

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    _TS as StatsForecastModel,
)
from statsforecast.models import (
    ADIDA as _ADIDA,
)
from statsforecast.models import (
    IMAPA as _IMAPA,
)
from statsforecast.models import (
    AutoARIMA as _AutoARIMA,
)
from statsforecast.models import (
    AutoCES as _AutoCES,
)
from statsforecast.models import (
    AutoETS as _AutoETS,
)
from statsforecast.models import (
    CrostonClassic as _CrostonClassic,
)
from statsforecast.models import (
    DynamicOptimizedTheta as _DOTheta,
)
from statsforecast.models import (
    HistoricAverage as _HistoricAverage,
)
from statsforecast.models import (
    SeasonalNaive as _SeasonalNaive,
)
from statsforecast.models import (
    Theta as _Theta,
)
from statsforecast.models import (
    ZeroModel as _ZeroModel,
)

from ..utils.forecaster import Forecaster, QuantileConverter, get_seasonality

os.environ["NIXTLA_ID_AS_COL"] = "true"


def run_statsforecast_model(
    model: StatsForecastModel,
    df: pd.DataFrame,
    h: int,
    freq: str,
    level: list[int | float] | None,
    quantiles: list[float] | None,
) -> pd.DataFrame:
    sf = StatsForecast(
        models=[model],
        freq=freq,
        n_jobs=-1,
        fallback_model=_SeasonalNaive(
            season_length=get_seasonality(freq),
        ),
    )
    qc = QuantileConverter(level=level, quantiles=quantiles)
    fcst_df = sf.forecast(df=df, h=h, level=qc.level)
    fcst_df = qc.maybe_convert_level_to_quantiles(
        df=fcst_df,
        models=[model.alias],
    )
    return fcst_df


class ADIDA(Forecaster):
    def __init__(
        self,
        alias: str = "ADIDA",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_ADIDA(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class AutoARIMA(Forecaster):
    def __init__(
        self,
        alias: str = "AutoARIMA",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_AutoARIMA(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class AutoCES(Forecaster):
    def __init__(
        self,
        alias: str = "AutoCES",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_AutoCES(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class AutoETS(Forecaster):
    def __init__(
        self,
        alias: str = "AutoETS",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_AutoETS(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class CrostonClassic(Forecaster):
    def __init__(
        self,
        alias: str = "CrostonClassic",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_CrostonClassic(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class DOTheta(Forecaster):
    def __init__(
        self,
        alias: str = "DOTheta",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_DOTheta(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class HistoricAverage(Forecaster):
    def __init__(
        self,
        alias: str = "HistoricAverage",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_HistoricAverage(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class IMAPA(Forecaster):
    def __init__(
        self,
        alias: str = "IMAPA",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_IMAPA(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class SeasonalNaive(Forecaster):
    def __init__(
        self,
        alias: str = "SeasonalNaive",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_SeasonalNaive(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class Theta(Forecaster):
    def __init__(
        self,
        alias: str = "Theta",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_Theta(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class ZeroModel(Forecaster):
    def __init__(
        self,
        alias: str = "ZeroModel",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_ZeroModel(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df
