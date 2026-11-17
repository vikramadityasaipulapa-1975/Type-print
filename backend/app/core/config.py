from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Typing Fingerprint API"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://typing:typing_dev@localhost:5432/typing_fingerprint"
    cors_origins: str = "http://localhost:5173"
    jwt_secret_key: str = ""
    jwt_expire_minutes: int = 30
    biometric_encryption_key: str = ""
    lstm_checkpoint_path: str = "models/keystroke_lstm.pt"
    verification_calibrator_path: str = "models/distance_calibrator.joblib"
    verification_threshold: float | None = None
    minimum_enrollment_samples: int = 5
    minimum_typing_events: int = 10
    model_metrics_path: str = "models/verification.metrics.json"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)
    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
settings = Settings()
