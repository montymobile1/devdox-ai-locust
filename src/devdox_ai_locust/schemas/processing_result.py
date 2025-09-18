from pydantic import BaseModel, field_validator, ValidationInfo
from typing import Optional, Type


class SwaggerProcessingRequest(BaseModel):
    swagger_url: Optional[str] = None

    @field_validator("swagger_url", mode="before")
    @classmethod
    def coerce_to_string(
        cls: Type["SwaggerProcessingRequest"], v: Optional[str]
    ) -> Optional[str]:
        if v is None:
            return v
        return str(v)

    @field_validator("swagger_url")
    @classmethod
    def validate_url_when_source_is_url(
        cls: Type["SwaggerProcessingRequest"], v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        if info.data.get("swagger_source") == "url" and not v:
            raise ValueError("swagger_url is required when source is url")
        return v
