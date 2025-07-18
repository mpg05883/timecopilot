from gluonts.torch.model.predictor import PyTorchPredictor
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from ..utils.gluonts_forecaster import GluonTSForecaster


class Moirai(GluonTSForecaster):
    def __init__(
        self,
        repo_id: str = "Salesforce/moirai-1.0-R-large",
        filename: str = "model.ckpt",
        context_length: int = 4096,
        patch_size: int = 32,
        num_samples: int = 100,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
        batch_size: int = 32,
        alias: str = "Moirai",
    ):
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            alias=alias,
            num_samples=num_samples,
        )
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.batch_size = batch_size

    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self.repo_id),
            prediction_length=prediction_length,
            context_length=self.context_length,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
        )
        predictor = model.create_predictor(batch_size=self.batch_size)
        return predictor
