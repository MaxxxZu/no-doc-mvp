from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # CORS
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed origins (comma-separated in ENV, parsed by pydantic as list)",
    )

    # models
    zsc_model: str = Field(default="NbAiLab/nb-bert-base-mnli", alias="ZSC_MODEL")
    ner_model: str = Field(default="Kushtrim/norbert3-large-ner", alias="NER_MODEL")

    # zero-shot params
    zsc_multi_label: bool = Field(default=True, alias="ZSC_MULTI_LABEL")
    zsc_max_len: int = Field(default=2048, alias="ZSC_MAX_LEN")

    # Hugging Face cache
    hf_home: Optional[str] = Field(default=None, alias="HF_HOME")
    transformers_cache: Optional[str] = Field(default=None, alias="TRANSFORMERS_CACHE")

    # server
    port: int = Field(default=8080, alias="PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
        populate_by_name=True,
    )


settings = Settings()
