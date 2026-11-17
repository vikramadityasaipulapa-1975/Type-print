from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.database.session import engine
router = APIRouter(prefix="/api", tags=["system"])
@router.get("/health")
def health() -> dict[str, str]:
    database = "unavailable"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database = "connected"
    except SQLAlchemyError:
        pass
    return {"status": "ok", "database": database}
