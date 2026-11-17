import json
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.database.session import get_db
from app.models.entities import ModelMetric, User
from app.schemas.api import ModelMetricsResponse, ModelStatusResponse
from app.services.biometric import ModelService, get_model_service

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status", response_model=ModelStatusResponse)
def model_status(user: User = Depends(get_current_user), service: ModelService = Depends(get_model_service)) -> dict[str, object]:
    return service.status()


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


@router.get("/metrics", response_model=ModelMetricsResponse)
def model_metrics(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ModelMetricsResponse:
    random_forest = _load_json(Path("models/random_forest.joblib.metrics.json"))
    verification = _load_json(Path(settings.model_metrics_path))
    reports = []
    if random_forest:
        test = random_forest.get("test", {})
        reports.append((str(random_forest.get("model_name", "random_forest")), None, None, None, test.get("classification_roc_auc_ovr_weighted")))
    if verification:
        test = verification.get("test_verification", {})
        reports.append(("lstm_embedding_verifier", test.get("far"), test.get("frr"), test.get("eer"), test.get("roc_auc")))
    for model_name, far, frr, eer, roc_auc in reports:
        row = db.scalar(select(ModelMetric).where(ModelMetric.model_name == model_name).order_by(ModelMetric.created_at.desc()).limit(1))
        if row is None:
            row = ModelMetric(model_name=model_name, far=far, frr=frr, eer=eer, roc_auc=roc_auc)
            db.add(row)
        else:
            row.far, row.frr, row.eer, row.roc_auc = far, frr, eer, roc_auc
    if reports:
        db.commit()
    return ModelMetricsResponse(
        random_forest=random_forest,
        verification=verification,
    )
