# Adapted from https://github.com/SalesforceAIResearch/gift-eval

import json
import math
import os
from collections.abc import Iterable, Iterator
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any
import numpy as np
import pickle
import pyarrow.compute as pc
from datasets import load_from_disk
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import get_seasonality, norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose
from gluonts.dataset.common import ListDataset

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "H": 48,  # Hourly
    "h": 48,
    "D": 14,  # Daily
    "d": 14,
    "W": 13,  # Weekly
    "w": 13,
    "M": 18,  # Monthly
    "m": 18,
    "ME": 18,  # End of month
    "Q": 8,  # Quarterly
    "q": 8,
    "QE": 8,  # End of quarter
    "A": 6,  # Annualy/yearly
    "y": 6,
    "YE": 6,  # End of year
}

PRED_LENGTH_MAP = {
    "S": 60,  # Seconds
    "s": 60,
    "T": 48,  # Minutely
    "min": 48,
    "H": 48,  # Hourly
    "h": 48,
    "D": 30,  # Daily
    "d": 30,
    "W": 8,  # Weekly
    "w": 8,
    "M": 12,  # Monthly
    "m": 12,
    "ME": 12,
    "Q": 8,  # Quarterly
    "q": 8,
    "QE": 8,
    "y": 6,  # Annualy/yearly
    "A": 6,
}

# Prediction lengths from TFB: https://arxiv.org/abs/2403.20150
TFB_PRED_LENGTH_MAP = {
    "U": 8,
    "T": 8,  # Minutely
    "H": 48,  # Hourly
    "h": 48,
    "D": 14,  # Daily
    "W": 13,  # Weekly
    "M": 18,  # Monthly
    "Q": 8,  # Quarterly
    "A": 6,  # Annualy/yearly
}

MIN_LENGTH = int(348 / 0.5)


class Term(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


class Domain(StrEnum):
    """
    Represents the dataset's domain.

    Attributes:
        CLIMATE: Datasets related to weather, climate, or environmental
            monitoring.
        CLOUDOPS: Datasets related to cloud infrastructure and operations.
        ECON_FIN: Economic and financial datasets.
        HEALTHCARE: Datasets from the healthcare domain.
        NATURE: Scientific or biological datasets.
        SALES: Datasets tracking retail or product sales.
        TRANSPORT: Datasets involving traffic or transportation.
        WEB: Datasets from web or online platforms.
        WEB_CLOUDOPS: A combined or hybrid domain covering both Web and
            CloudOps.
        ALL: Represents all domains combined.
    """

    CLIMATE = "Climate"  # This's only in the pretrain split
    CLOUDOPS = "CloudOps"  # This's only in the pretrain split
    ECON_FIN = "Econ/Fin"
    HEALTHCARE = "Healthcare"
    NATURE = "Nature"
    SALES = "Sales"
    TRANSPORT = "Transport"
    WEB = "Web"  # This's only in the pretrain split
    WEB_CLOUDOPS = "Web/CloudOps"  # This's only in the train-test split
    ENERGY = "Energy"
    ALL = "All"


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


def maybe_reconvert_freq(freq: str) -> str:
    """if the freq is one of the newest pandas freqs, convert it to the old freq"""
    deprecated_map = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    if freq in deprecated_map:
        return deprecated_map[freq]
    return freq


class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


class Dataset:
    def _storage_path_from_env_var(self, env_var: str) -> Path:
        load_dotenv()
        env_var_value = os.getenv(env_var)
        if env_var_value is None:
            raise ValueError(f"Environment variable {env_var} is not set")
        return Path(env_var_value)

    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = True,
        storage_path: Path | str | None = None,
        storage_env_var: str = "GIFT_EVAL",
        stl_cache: str="stl_cache"  
    ):
        self.term = Term(term)
        self.name = name
        self.stl_cache = stl_cache
        if storage_path is None:
            storage_path = self._storage_path_from_env_var(storage_env_var)
        else:
            storage_path = Path(storage_path)

        self.dataset_directory = os.getenv(storage_env_var)
        self.hf_dataset = load_from_disk(self.storage_path).with_format("numpy")

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(
            compose(process, itemize_start),
            self.hf_dataset,
        )

        # Automatically convert multivariate datasets to univariate
        if to_univariate and self.target_dim > 1:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )
        
        self.stl_gluonts_dataset = self._insert_stl_components() if self.stl_cache else self.gluonts_dataset
            

    def _load_stl_components(self, path: Path) -> list[np.ndarray]:
        with open(path, "rb") as file:
            components = pickle.load(file)
        return [
            np.array(c) if not isinstance(c, np.ndarray) else c
            for c in components
        ]
        
    def _insert_stl_components(self) -> ListDataset:
        trend_list = self._load_stl_components(self.trend_path)
        seasonal_list = self._load_stl_components(self.seasonal_path)
        residual_list = self._load_stl_components(self.residual_path)
        
        zipped = zip(
            self.gluonts_dataset, 
            trend_list, 
            seasonal_list, 
            residual_list,
        )
        entries = []
        
        for entry, trend, seasonal, residual in zipped:
            if len(trend) < MIN_LENGTH:
                n_periods = MIN_LENGTH - len(trend)

                # Adjust the start time of the entry
                entry["start"] = entry["start"] - n_periods  

                entry["target"] = np.pad(
                    entry["target"], (MIN_LENGTH - len(trend), 0)
                )
                trend = np.pad(trend, (MIN_LENGTH - len(trend), 0))
                seasonal = np.pad(seasonal, (MIN_LENGTH - len(seasonal), 0))
                residual = np.pad(residual, (MIN_LENGTH - len(residual), 0))
                
            entries.append(
                {
                    **entry,
                    "trend": trend,
                    "seasonal": seasonal,
                    "residual": residual,
                }
            )
        return ListDataset(entries, freq=self.freq)
        
            
    @property
    def trend_path(self) -> Path:
        return (Path(self.stl_cache) / self.config / "trend").with_suffix(".pk")
    
    @property
    def seasonal_path(self) -> Path:
        return (Path(self.stl_cache) / self.config / "seasonal").with_suffix(".pk")
    
    @property
    def residual_path(self) -> Path:
        return (Path(self.stl_cache) / self.config / "residual").with_suffix(".pk")
        

    @cached_property
    def prediction_length(self) -> int:
        freq = norm_freq_str(to_offset(self.freq).name)
        freq = maybe_reconvert_freq(freq)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        return (
            target.shape[0]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        elif (
            len(
                (
                    past_feat_dynamic_real := self.hf_dataset[0][
                        "past_feat_dynamic_real"
                    ]
                ).shape
            )
            > 1
        ):
            return past_feat_dynamic_real.shape[0]
        else:
            return 1

    @cached_property
    def windows(self) -> int:
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column("target"), 0, 1)
                )
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(self.hf_dataset.data.column("target"))
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.stl_gluonts_dataset, offset=-self.prediction_length * (self.windows + 1)
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.stl_gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.stl_gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data

    @property
    def config(self) -> str:
        """
        Returns the dataset's configuration formatted as `name`/`freq`/`term`.
        This's is used for formatting dataset names and terms in results files.
        - `name` is the dataset's name in lowercase, with some additional
            formatting for specific datasets.
            - E.g. "saugeenday" is formatted as "saugeen".
        - `freq` is the dataset's frequency with the optional dash removed.
            - E.g. "W-SUN" is formatted as "W".
        - `term` is the dataset's term (short, medium, long).

        Returns:
            str: The dataset's configuration formatted as `name`/`freq`/`term`.
        """
        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }
        name = self.name.split("/")[0] if "/" in self.name else self.name
        cleaned_name = pretty_names.get(name.lower(), name.lower())
        cleaned_freq = self.freq.split("-")[0]
        return f"{cleaned_name}/{cleaned_freq}/{self.term}"

    @property
    def cleaned_config(self) -> str:
        """
        Returns the dataset's configuration formatted as `name`_`freq`_`term`
        for storing things e.g. W&B runs.
        """
        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }
        name = self.name.split("/")[0] if "/" in self.name else self.name
        cleaned_name = pretty_names.get(name.lower(), name.lower())
        cleaned_freq = self.freq.split("-")[0]
        return f"{cleaned_name}_{cleaned_freq}_{self.term}"

    @property
    def seasonality(self) -> int:
        """
        Computes the dataset's seasonality (number of time steps in one
        seasonal cycle). This's a thin wrapper around GluonTS's
        `get_seasonality`.

        Returns:
            int: The dataset's seasonality.
        """
        return get_seasonality(self.freq)

    @property
    def gift_split(self) -> str:
        """
        Returns "train_test" if the dataset is in the GIFT-Eval's train-test
        split. Else, retuns "pretrain".
        """
        file_path = Path(self.dataset_directory) / "train_test" / self.name
        return "train_test" if file_path.exists() else "pretrain"

    @property
    def storage_path(self) -> str:
        """
        Returns a path to where the dataset's stored on disk using the root
        directory specified by the storage enviornment variable.
        """
        return str(Path(self.dataset_directory) / self.gift_split / self.name)

    @cached_property
    def metadata(self) -> dict[str, Any]:
        path = Path(__file__).parent / "meta" / self.gift_split / "metadata.json"
        with open(path) as file:
            metadata = json.load(file)
        return metadata[self.name]

    @property
    def domain(self) -> Domain:
        """
        Returns the dataset's domain.
        """
        return Domain(self.metadata["domain"])
