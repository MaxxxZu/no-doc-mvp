import json
from typing import List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict
from pydantic_settings.sources import SettingsError


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1, le=65535, validation_alias=AliasChoices("PORT", "NDM_PORT"))

    cors_origins: List[str] = Field(
        default_factory=list,
        description="Allowed origins as a JSON array of strings",
    )

    zsc_model: str = Field(
        "NbAiLab/nb-bert-base-mnli",
        validation_alias=AliasChoices("NDM_ZSC_MODEL", "ZSC_MODEL"),
    )
    ner_model: str = Field(
        "Kushtrim/norbert3-large-ner",
        validation_alias=AliasChoices("NDM_NER_MODEL", "NER_MODEL"),
    )
    zsc_multi_label: bool = Field(
        True, validation_alias=AliasChoices("NDM_ZSC_MULTI_LABEL", "ZSC_MULTI_LABEL")
    )
    zsc_max_len: int = Field(
        2048, gt=0, validation_alias=AliasChoices("NDM_ZSC_MAX_LEN", "ZSC_MAX_LEN")
    )
    hf_home: Optional[str] = Field(
        None, validation_alias=AliasChoices("NDM_HF_HOME", "HF_HOME")
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NDM_",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @field_validator("zsc_model", "ner_model")
    @classmethod
    def nonempty_model_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model ID must not be blank")
        return value

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        def canonicalize(values):
            result = {}
            for name, field in settings_cls.model_fields.items():
                alias = field.validation_alias
                keys = alias.choices if isinstance(alias, AliasChoices) else [name]
                for key in [*keys, name]:
                    if key in values:
                        result[name] = values[key]
                        break
            return result

        def read_env(source):
            raw, _, _ = source.get_field_value(
                settings_cls.model_fields["cors_origins"], "cors_origins"
            )
            if raw is not None:
                try:
                    origins = json.loads(raw)
                except ValueError as exc:
                    raise SettingsError("NDM_CORS_ORIGINS must be a JSON array") from exc
                if not isinstance(origins, list):
                    # In particular, do not let decoded null disappear into defaults.
                    raise SettingsError("NDM_CORS_ORIGINS must be a JSON array")
            # DotEnvSettingsSource normally includes unrelated/unused dotenv keys.
            # Use only declared fields, and merge sources by field rather than alias:
            # process NDM_PORT must beat dotenv PORT despite PORT's alias priority.
            return canonicalize(EnvSettingsSource.__call__(source))

        return (
            lambda: canonicalize(init_settings()),
            lambda: read_env(env_settings),
            lambda: read_env(dotenv_settings),
        )


settings = Settings()
