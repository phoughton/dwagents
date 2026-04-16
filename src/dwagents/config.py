from pydantic_settings import BaseSettings


class DoublewordSettings(BaseSettings):
    model_config = {"env_prefix": "DOUBLEWORD_"}

    api_key: str = ""
    base_url: str = "https://api.doubleword.ai/v1/"
    model: str = "Qwen/Qwen3.5-397B-A17B-FP8"
    batch_window_seconds: float = 10.0
    batch_size: int = 1000
    poll_interval_seconds: float = 5.0
    completion_window: str = "1h"


settings = DoublewordSettings()
