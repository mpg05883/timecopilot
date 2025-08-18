from typing import Literal

import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.split import TrainingDataset
from gluonts.torch.batchify import stack
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    # TFTInstanceSplitter
)

from utils.tabpfn_features import DefaultFeatures


class InferenceSplitter(InstanceSplitter):
    def flatmap_transform(self, data, is_train):
        # force it to behave like training split, so it returns future_… fields
        yield from super().flatmap_transform(data, is_train=True)


def custom_batchify(data, device=None):
    """
    samples: list of dicts, each dict has keys like
      "target", FieldName.OBSERVED_VALUES, "trend", "seasonal", "residual", etc.
    device: torch device or None
    """
    # keys we DON'T want in the final batch
    skip_keys = {"start", "item_id", "freq", "past_feat_dynamic_real"}
    # import pdb; pdb.set_trace()
    batch = {
        key: stack([item[key] for item in data], device=device)
        for key in data[0].keys()
        if key not in skip_keys
    }
    return batch


TIME_SERIES_FIELDS = [
    FieldName.OBSERVED_VALUES,
    "trend",
    "seasonal",
    "residual",
]

import itertools

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

MAX_EMPTY_YIELDS = 100


# ─── helper to limit the number of batches ────────────────────────────────────
class LimitedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, max_batches: int):
        self.dataset = dataset
        self.max_batches = max_batches

    def __iter__(self):
        return itertools.islice(self.dataset, self.max_batches)


# from gluonts.transform import create_mask_unobserved_transformation
from collections.abc import Iterable

import pandas as pd
from gluonts.dataset.common import DataEntry


class AddTimestamps(Transformation):
    """
    Reads 'start_field' + length of 'target_field' to build a raw
    numpy.datetime64[ns] array and writes it to 'output_field'.
    """

    def __init__(self, start_field: str, target_field: str, output_field: str):
        self.start_field = start_field
        self.target_field = target_field
        self.output_field = output_field

    def _safe_period_to_timestamp(self, length, freq):
        """Convert PeriodIndex to timestamps safely, handling overflows"""
        # For very long series, create a synthetic datetime index
        # that preserves calendar features without actual overflow

        # Option A: Use a safe date range within pandas limits
        safe_start = pd.Timestamp("1900-01-01")
        if freq in ["A-DEC", "A", "Y", "YS"]:
            # For yearly data, create years 1900-2399 cyclically if needed
            years_available = 2260 - 1900  # ~360 years
            if length <= years_available:
                return pd.date_range(start=safe_start, periods=length, freq=freq)
            else:
                # Cycle through available years
                full_cycles = length // years_available
                remainder = length % years_available

                idx_parts = []
                for cycle in range(full_cycles):
                    cycle_idx = pd.date_range(
                        start=safe_start, periods=years_available, freq=freq
                    )
                    idx_parts.append(cycle_idx)

                if remainder > 0:
                    remainder_idx = pd.date_range(
                        start=safe_start, periods=remainder, freq=freq
                    )
                    idx_parts.append(remainder_idx)

                return pd.DatetimeIndex(
                    np.concatenate([idx.values for idx in idx_parts])
                )
        else:
            # For non-yearly frequencies, use original approach with safety limit
            safe_length = min(length, self.max_years * 365)  # Rough safety limit
            return pd.date_range(start=safe_start, periods=safe_length, freq=freq)

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterable[DataEntry]:
        for entry in data_it:
            start = entry[self.start_field]
            length = len(entry[self.target_field])
            freq = entry.get("freq", entry.get("frequency"))

            # Use new aliases for frequency strings
            if "W" not in freq:
                freq = freq.replace("T", "min")  #
                freq = freq.replace("H", "h")

            # handle Period vs Timestamp
            if isinstance(start, pd.Period):
                # Clip `start_ts` to the minimum allowed timestamp
                try:
                    start_ts = start.to_timestamp(how="start")
                except Exception:
                    start_ts = pd.Timestamp.min
            else:
                start_ts = pd.to_datetime(start, errors="coerce")

            # Ensure `start_ts` is within the valid range
            if start_ts < pd.Timestamp.min:
                start_ts = pd.Timestamp.min
            elif start_ts > pd.Timestamp.max:
                start_ts = pd.Timestamp.max

            # Handle cases when the length exceeds the max timestamp
            try:
                idx = pd.date_range(start=start_ts, periods=length, freq=freq)
            except (pd._libs.tslibs.np_datetime.OutOfBoundsDatetime, OverflowError):
                idx = self._safe_period_to_timestamp(
                    length,
                    freq,
                )

            times = np.array(idx.values, dtype="datetime64[ns]")
            time_pds = pd.DataFrame(times, columns=["timestamp"])
            calender_feat = DefaultFeatures.add_calendar_features(time_pds)
            indext_feat = DefaultFeatures.add_running_index(time_pds)
            tsdf = pd.concat([indext_feat, calender_feat], axis=1)
            entry[self.output_field] = tsdf.values.T.astype("float32")
            yield entry


class ProbabilisticMergedIterableDataset(IterableDataset):
    def __init__(
        self,
        datasets,
        probabilities,
        transform,
        is_train: bool = False,
        resample_exhausted: bool = False,
    ):
        """
        Args:
            datasets: list of GluonTS TrainingDataset (or any iterable dataset)
            probabilities: list of probabilities (float), same length as datasets
            transform: GluonTS transform (e.g., mask_unobserved + InstanceSplitter)
        """
        assert len(datasets) == len(probabilities)
        self.datasets = datasets
        self.probabilities = np.array(probabilities, dtype=np.float64)
        self.probabilities /= self.probabilities.sum()
        self.transform = transform
        self.is_train = is_train
        self.resample_exhausted = resample_exhausted

    def __iter__(self):
        #  lazily apply transform to each dataset
        gens = [
            lambda d=ds: self.transform(d, is_train=self.is_train)
            for ds in self.datasets
        ]
        worker_info = get_worker_info()
        if worker_info is not None:
            wid, nw = worker_info.id, worker_info.num_workers
            gens = gens[wid::nw]
            probs = self.probabilities[wid::nw]
        else:
            probs = self.probabilities
        probs = probs / probs.sum()

        # build iterators, optionally cycling exhausted streams
        if self.resample_exhausted:
            iterators = [itertools.cycle(g()) for g in gens]
        else:
            iterators = [iter(g()) for g in gens]

        while iterators:
            idx = np.random.choice(len(iterators), p=probs)
            try:
                yield next(iterators[idx])
            except StopIteration:
                # remove exhausted stream unless cycling
                if self.resample_exhausted:
                    continue
                iterators.pop(idx)
                probs = np.delete(probs, idx)
                if len(probs) == 0:
                    break
                probs = probs / probs.sum()


def create_mask_unobserved_transformation():
    # Impute "target" field
    mask_unobserved_target = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )

    # Impute "trend" field
    mask_unobserved_trend = AddObservedValuesIndicator(
        target_field="trend",
        output_field=FieldName.OBSERVED_VALUES,
    )

    # Impute "seasonal" field
    mask_unobserved_seasonal = AddObservedValuesIndicator(
        target_field="seasonal",
        output_field=FieldName.OBSERVED_VALUES,
    )

    # Impute "residual" field
    mask_unobserved_residual = AddObservedValuesIndicator(
        target_field="residual",
        output_field=FieldName.OBSERVED_VALUES,
    )

    # mask_unobserved_time_feat = AddObservedValuesIndicator(
    #     target_field=FieldName.FEAT_TIME,
    #     output_field=FieldName.OBSERVED_VALUES,
    # )

    # mask_unobserved_past_time_feat = AddObservedValuesIndicator(
    #     target_field="past_time_feat",
    #     output_field=FieldName.OBSERVED_VALUES,
    # )

    # mask_unobserved_future_time_feat = AddObservedValuesIndicator(
    #     target_field="future_time_feat",
    #     output_field=FieldName.OBSERVED_VALUES,
    # )

    # Combine all the mask transformations into one transformation
    mask_unobserved = (
        mask_unobserved_target
        + mask_unobserved_trend
        + mask_unobserved_seasonal
        + mask_unobserved_residual
        # + mask_unobserved_time_feat
        # + mask_unobserved_past_time_feat
        # + mask_unobserved_future_time_feat
    )

    return mask_unobserved


# Custom function to create a cyclic dataloader
def prepare_cyclic_dataloader(
    dataset: TrainingDataset,
    split: Literal["training", "validation"],
    context_length: int,
    prediction_length: int,
    num_batches_per_epoch: int,
    batch_size: int,
):
    from gluonts.itertools import Cyclic, IterableSlice, Map, PseudoShuffled

    # Create a list of dataloaders, one for each dataset

    # Create instance sampler
    instance_sampler = ExpectedNumInstanceSampler(
        num_instances=1,
        min_future=prediction_length,
    )

    # Initialize a splitter
    splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=TIME_SERIES_FIELDS,
    )

    # Create a transformation
    mask_unobserved = create_mask_unobserved_transformation()

    # Create dataloader for this dataset
    if split == "training":
        loader = TrainDataLoader(
            dataset,
            batch_size=batch_size,
            stack_fn=custom_batchify,
            transform=mask_unobserved + splitter,
        )
    else:
        loader = ValidationDataLoader(
            dataset,
            batch_size=batch_size,
            stack_fn=custom_batchify,
            transform=mask_unobserved + splitter,
        )

    iterator = iter(loader)
    shuffle_buffer_length = min(1000, max(100, batch_size * 10))
    cyclic_iter = Cyclic(
        PseudoShuffled(iterator, shuffle_buffer_length=shuffle_buffer_length)
    )

    # cyclic_iter = Cyclic(PseudoShuffled(iter(loader)))

    # Create an infinite iterator that yields the specified number of batches per epoch
    return Map(lambda x: x, IterableSlice(cyclic_iter, num_batches_per_epoch))


def prepare_dataloader(
    dataset: TrainingDataset,
    split: Literal["training", "validation"],
    context_length: int,
    prediction_length: int,
    num_batches_per_epoch: int,
    batch_size: int,
    use_time_features: bool = False,
) -> TrainDataLoader | ValidationDataLoader:
    # On average, num_instances time points will be sampled per time series
    instance_sampler = ExpectedNumInstanceSampler(
        num_instances=1,
        min_future=prediction_length,
    )
    pad = PadShortSeries(context_length + prediction_length + 1, pad_value=np.nan)

    if use_time_features:
        add_timestamps = AddTimestamps(
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,  # or "time_feat", or "time_idx", any name you like
        )

        # ─── 2) Create instance sampler and splitter ─────────────────────────────────
        splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=TIME_SERIES_FIELDS + [FieldName.FEAT_TIME],
            # time_series_fields=TIME_SERIES_FIELDS,
        )
        transform = (
            pad + add_timestamps + create_mask_unobserved_transformation() + splitter
        )  #

    else:
        # ─── 2) Create instance sampler and splitter ─────────────────────────────────
        splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=TIME_SERIES_FIELDS,
        )
        transform = pad + create_mask_unobserved_transformation() + splitter

    if split == "training":
        return TrainDataLoader(
            dataset,
            batch_size=batch_size,
            stack_fn=custom_batchify,
            transform=transform,
            num_batches_per_epoch=num_batches_per_epoch,
        )
    else:
        return ValidationDataLoader(
            dataset,
            batch_size=batch_size,
            stack_fn=custom_batchify,
            transform=transform,
        )


from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.transform import FilterTransformation


class PadShortSeries(Transformation):
    def __init__(self, min_length: int, pad_value: float = 0.0):
        super().__init__()  # call base init
        self.min_length = min_length
        self.pad_value = pad_value

    def __call__(self, data_iter, is_train: bool):
        """
        Make the class itself callable over an iterable of entries.
        """
        for entry in data_iter:
            yield self.apply(entry, is_train)

    def apply(self, entry: dict, is_train: bool) -> dict:
        """
        Pads a too‑short 'target' up to min_length by prepending pad_value,
        and shifts the 'start' timestamp accordingly.
        """
        target = entry["target"]
        L = len(target)
        # import pdb; pdb.set_trace()

        if self.min_length > L:
            pad_width = self.min_length - L
            # prepend pad_value
            entry["target"] = np.concatenate(
                [np.full(pad_width, self.pad_value, dtype=target.dtype), target]
            )
            entry["trend"] = np.concatenate(
                [np.full(pad_width, self.pad_value, dtype=target.dtype), target]
            )
            entry["seasonal"] = np.concatenate(
                [np.full(pad_width, self.pad_value, dtype=target.dtype), target]
            )
            # import pdb; pdb.set_trace()
            # entry["time_feat"] = np.concatenate(
            #     [np.full(pad_width, self.pad_value, dtype=target.dtype), target]
            # )
            entry["residual"] = np.concatenate(
                [np.full(pad_width, self.pad_value, dtype=target.dtype), target]
            )

            # # shift the start timestamp back by pad_width steps
            # start_period = pd.Period(entry["start"], freq=entry["freq"])
            # entry["start"] = (start_period - pad_width).to_timestamp()

        return entry


def has_any_valid_series(datasets, min_length):
    # scan until you find one
    for ds in datasets:
        for entry in ds:
            target = entry["target"]
            if isinstance(target, np.ndarray) and len(target) >= min_length:
                return True
    return False


def prepare_dataloader_multiple(
    datasets: list,
    split: Literal["training", "validation"],
    context_length: int,
    prediction_length: int,
    num_batches_per_epoch: int,
    batch_size: int,
    probabilities: list[float] | None = None,
    seed: int | None = 2025,  # for reproducibility
    cycle_exhausted: bool = True,  # resample small datasets
    limit_validation_batches: int | None = 100,  # cap val batches
    use_time_features: bool = False,
):
    assert len(datasets) > 0
    # ─── reproducibility ────────────────────────────────────────────────────────
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if probabilities is None:
        probabilities = [1.0 / len(datasets)] * len(datasets)

    # ─── 1) Filter out entire datasets with NO valid samples ────────────────────
    # 1) filter entries _within_ each dataset
    filtered_datasets = []
    filtered_probs = []
    min_len = prediction_length
    use_length_weights = True
    for ds, p in zip(datasets, probabilities, strict=False):
        # pull out only the entries that are long enough
        entries = [
            entry
            for entry in ds
            if isinstance(entry["target"], np.ndarray)
            and len(entry["target"]) >= min_len
        ]
        if len(entries) > 0:
            # rebuild a ListDataset with the same freq
            filtered_datasets.append(ListDataset(entries, freq=entries[0]["freq"]))
            filtered_probs.append(p)

    total_p = sum(filtered_probs)
    probabilities = [p / total_p for p in filtered_probs]
    datasets = filtered_datasets

    if split == "training":
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=1, min_future=prediction_length
        )
    else:
        instance_sampler = ValidationSplitSampler(min_future=prediction_length)

    def debug_filter(entry):
        target = entry["target"]
        # safe‐guard: only try to print length if it’s actually an ndarray
        if isinstance(target, np.ndarray):
            L = len(target)
            print(f"  ▶ target length = {L}")
            print(prediction_length < L)
        else:
            # print(f"  ▶ target is not ndarray, but {type(target)}")
            return False

        # now return the real predicate
        return prediction_length < L  # (context_length + prediction_length)

    filter_valid = FilterTransformation(debug_filter)

    # add_time_features = AddTimeFeatures(
    #     start_field=FieldName.START,
    #     target_field=FieldName.TARGET,
    #     output_field=FieldName.FEAT_TIME, #FieldName.FEAT_TIME,       # ← note: FEAT_TIME, not PAST_TIME_FEAT
    #     time_features=[
    #         hour_of_day,    # callable: index.hour / 23.0 - 0.5
    #         day_of_week,    # callable: index.dayofweek / 6.0 - 0.5
    #         month_of_year,  # callable: (index.month-1)/11.0 - 0.5
    #     ],
    #     pred_length=prediction_length,
    # )
    pad = PadShortSeries(context_length + prediction_length + 1, pad_value=np.nan)

    if use_time_features:
        add_timestamps = AddTimestamps(
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,  # or "time_feat", or "time_idx", any name you like
        )

        # ─── 2) Create instance sampler and splitter ─────────────────────────────────
        splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=TIME_SERIES_FIELDS + [FieldName.FEAT_TIME],
            # time_series_fields=TIME_SERIES_FIELDS,
        )
        transform = (
            add_timestamps + pad + create_mask_unobserved_transformation() + splitter
        )  #

    else:
        # ─── 2) Create instance sampler and splitter ─────────────────────────────────
        splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=TIME_SERIES_FIELDS,
        )
        transform = pad + create_mask_unobserved_transformation() + splitter

    # transform = filter_valid + create_mask_unobserved_transformation() + splitter
    # transform = fill_defaults + pad + create_mask_unobserved_transformation() + splitter
    # transform = add_time_features + pad + create_mask_unobserved_transformation()+ splitter #
    # transform = filter_valid + create_mask_unobserved_transformation() + splitter
    # transform = create_mask_unobserved_transformation() + splitter

    # ─── build merged iterable ─────────────────────────────────────────────────
    merged = ProbabilisticMergedIterableDataset(
        datasets=datasets,
        probabilities=probabilities,
        transform=transform,
        is_train=(split == "training"),
        resample_exhausted=cycle_exhausted,
    )

    if split == "training":
        return TrainDataLoader(
            merged,
            batch_size=batch_size,
            stack_fn=custom_batchify,
            num_batches_per_epoch=num_batches_per_epoch,
        )
    else:
        # optionally limit the number of validation batches
        if limit_validation_batches is not None:
            merged = LimitedIterableDataset(merged, limit_validation_batches)
        return ValidationDataLoader(
            merged,
            batch_size=batch_size,
            stack_fn=custom_batchify,
        )


def get_input_transform(
    context_length: int,
    prediction_length: int,
    use_time_features=False,
):

    pad = PadShortSeries(
        context_length + prediction_length + 1,
        pad_value=np.nan,
    )
    # add_time_features = AddTimeFeatures(
    mask_unobserved = create_mask_unobserved_transformation()

    if not use_time_features:
        prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=TIME_SERIES_FIELDS,
        )

        return pad + mask_unobserved + prediction_splitter
    else:
        FieldName.TREND = "trend"
        add_timestamps = AddTimestamps(
            start_field=FieldName.START,
            target_field=FieldName.TREND,
            output_field=FieldName.FEAT_TIME,
        )

        prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=TIME_SERIES_FIELDS + [FieldName.FEAT_TIME],
        )

        return pad + add_timestamps + mask_unobserved + prediction_splitter
