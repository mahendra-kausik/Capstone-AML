from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TopFeature(BaseModel):
    name: str
    index: int | None = None
    contribution: float | None = None
    shap_value: float | None = None
    feature_value: float | None = None


class PredictRequest(BaseModel):
    model: Literal["static", "evolve"] = "static"
    tx_id: str = "unknown"
    time_step: int | None = None
    features: list[float] | None = None
    feature_dict: dict[str, float] | None = None

    @model_validator(mode="after")
    def check_features(self):
        if not self.features and not self.feature_dict:
            raise ValueError("Provide features (165) or feature_dict")
        if self.features and len(self.features) != 165:
            raise ValueError("features must have length 165")
        return self


class PredictResponse(BaseModel):
    tx_id: str
    model: str
    risk_score: float
    prediction: str
    confidence: float
    prob_licit: float
    prob_illicit: float
    top_features: list[TopFeature] = []
    prediction_id: str | None = None


class TransactionInput(BaseModel):
    tx_id: str
    time_step: int | None = None
    features: list[float] | None = None
    feature_dict: dict[str, float] | None = None


class EdgeInput(BaseModel):
    src_tx_id: str
    dst_tx_id: str


class BatchPredictRequest(BaseModel):
    model: Literal["static", "evolve"] = "static"
    transactions: list[TransactionInput]
    edges: list[EdgeInput] | None = None


class BatchPredictResponse(BaseModel):
    job_id: str
    count: int
    results: list[PredictResponse]
    summary: dict[str, Any]


class ExplainRequest(PredictRequest):
    nsamples: int | None = Field(default=None, ge=10, le=500)


class ExplainResponse(BaseModel):
    tx_id: str
    model: str
    risk_score: float
    prediction: str
    confidence: float
    top_features: list[TopFeature]
    method: str = "kernel_shap"
    shap_id: str | None = None
