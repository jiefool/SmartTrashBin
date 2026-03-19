from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
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

    # GPIO – HC-SR04 ultrasonic sensor
    gpio_echo_pin: int = 23
    gpio_trig_pin: int = 24

    # Auto-detection (sensor-triggered classification)
    detection_threshold_inches: float = 5.0      # classify when object is 0–5 inches
    detection_cooldown_seconds: float = 5.0      # seconds between classifications
    detection_poll_interval: float = 0.3         # how often to read sensor (seconds)

    # Classification model
    model_path: str = "data/models/trashnet_mobilenetv2.tflite"



settings = Settings()

