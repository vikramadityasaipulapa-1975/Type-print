from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.analytics import router as analytics_router
from app.api.auth import router as auth_router
from app.api.biometrics import router as biometrics_router
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.typing import router as typing_router
from app.core.config import settings
app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True, allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Authorization", "Content-Type"])
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(biometrics_router)
app.include_router(typing_router)
app.include_router(analytics_router)
app.include_router(models_router)
@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "docs": "/docs"}
