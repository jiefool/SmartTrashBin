from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "SmartTrashBin"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "DEBUG"

    # Trash Bin
    bin_capacity_cm: float = 100.0
    alert_threshold_percent: float = 80.0
    poll_interval_seconds: int = 10

    # Classification model
    model_path: str = "data/models/trashnet_mobilenetv2.keras"



settings = Settings()

