import gluonts.time_feature
import numpy as np
import pandas as pd


class DefaultFeatures:
    @staticmethod
    def add_running_index(df: pd.DataFrame) -> pd.Series:
        # df["running_index"] = range(len(df))
        # # Normalize running index to be between 0 and 1
        # df["running_index"] = df["running_index"] / (len(df) - 1)
        df["running_index"] = np.linspace(0.0, 1.0, num=len(df), dtype="float32")
        return df

    @staticmethod
    def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        CALENDAR_COMPONENT = [
            "year",
            # "month",
            # "day",
        ]

        CALENDAR_FEATURES = [
            # (feature, natural seasonality)
            ("hour_of_day", 24),
            ("day_of_week", 7),
            ("day_of_month", 30.5),
            ("day_of_year", 365),
            ("week_of_year", 52),
            ("month_of_year", 12),
        ]
        # import pdb; pdb.set_trace()
        df.set_index("timestamp", inplace=True)
        timestamps = df.index.get_level_values("timestamp")
        # timestamps = df["timestamp"].dt.to_pydatetime()

        for component_name in CALENDAR_COMPONENT:
            df[component_name] = getattr(timestamps, component_name)

        for feature_name, seasonality in CALENDAR_FEATURES:
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)
            if seasonality is not None:
                df[f"{feature_name}_sin"] = np.sin(
                    2 * np.pi * feature / (seasonality - 1)
                )  # seasonality - 1 because the value starts from 0
                df[f"{feature_name}_cos"] = np.cos(
                    2 * np.pi * feature / (seasonality - 1)
                )
            else:
                df[feature_name] = feature
        df["year"] = df["year"] * 0.001
        return df
