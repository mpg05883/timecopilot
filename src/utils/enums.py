from enum import StrEnum


class Term(StrEnum):
    """
    Represents the forecasting horizon category.

    Attributes:
        SHORT: Short-term forecasting.
        MEDIUM: Medium-term forecasting.
        LONG: Long-term forecasting.

    Properties:
        prediction_length (int): Forecast horizon length.
        context_length (int): Input window length.
        multiplier (int): Multiplier used to scale prediction length.
    """

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def prediction_length(self) -> int:
        return {
            Term.SHORT: 48,
            Term.MEDIUM: 480,
            Term.LONG: 720,
        }[self]

    @property
    def context_length(self) -> int:
        return {
            Term.SHORT: 336,
            Term.MEDIUM: 336,
            Term.LONG: 336,
        }[self]

    @property
    def multiplier(self) -> int:
        return {
            Term.SHORT: 1,
            Term.MEDIUM: 10,
            Term.LONG: 15,
        }[self]


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

    CLIMATE = "Climate"  # Only in the pretrain split
    CLOUDOPS = "CloudOps"  # Only in the pretrain split
    ECON_FIN = "Econ/Fin"
    HEALTHCARE = "Healthcare"
    NATURE = "Nature"
    SALES = "Sales"
    TRANSPORT = "Transport"
    WEB = "Web"  # Only in the pretrain split
    WEB_CLOUDOPS = "Web/CloudOps"  # Only in the train-test split
    ENERGY = "Energy"
    ALL = "All"
