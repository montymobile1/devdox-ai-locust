from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional


class SwaggerProcessingRequest(BaseModel):
    swagger_url: Optional[str] = None

    @field_validator("swagger_url", mode="before")
    @classmethod
    def coerce_to_string(cls, v):
        if v is None:
            return v
        return str(v)


    @field_validator('swagger_url')
    @classmethod
    def validate_url_when_source_is_url(cls, v, info: ValidationInfo):
        # Access other field values through info.data
        if info.data.get('swagger_source') == 'url' and not v:
            raise ValueError('swagger_url is required when source is url')
        return v
