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

    # GPIO – Relays (1 pin each)
    relay_1_pin: int = 2         # Biodegradable
    relay_2_pin: int = 3         # Non-Biodegradable
    relay_3_pin: int = 4         # Hazardous
    relay_on_seconds: float = 10.0  # how long relay stays ON

    # GPIO – Bin level ultrasonic sensors (trig / echo per bin)
    bin_sensor_1_trig: int = 10    # Biodegradable
    bin_sensor_1_echo: int = 9
    bin_sensor_2_trig: int = 5     # Non-Biodegradable
    bin_sensor_2_echo: int = 11
    bin_sensor_3_trig: int = 6     # Hazardous
    bin_sensor_3_echo: int = 13
    bin_full_threshold_inches: float = 12.0  # bin full when ≤12 inches
    bin_level_poll_interval: float = 1.0     # how often to check (seconds)

    # GPIO – LEDs (bin full indicators)
    led_1_pin: int = 17           # Biodegradable
    led_2_pin: int = 27           # Non-Biodegradable
    led_3_pin: int = 22           # Hazardous

    # Classification model
    model_path: str = "data/models/trashnet_mobilenetv2.tflite"



settings = Settings()

