from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Type


class SwaggerProcessingRequest(BaseModel):
    """Request model for processing an OpenAPI/Swagger schema.

    Exactly one of ``swagger_url`` or ``swagger_file_path`` must be provided.
    Supplying both or neither raises a ``ValueError`` at construction time.
    """

    swagger_url: Optional[str] = None
    swagger_file_path: Optional[str] = None

    @field_validator("swagger_url", "swagger_file_path", mode="before")
    @classmethod
    def coerce_to_string(
        cls: Type["SwaggerProcessingRequest"], v: Optional[str]
    ) -> Optional[str]:
        if v is None:
            return v
        return str(v)

    @model_validator(mode="after")
    def validate_single_source(self) -> "SwaggerProcessingRequest":
        """Exactly one of swagger_url or swagger_file_path must be provided."""
        has_url = bool(self.swagger_url)
        has_file = bool(self.swagger_file_path)
        if has_url and has_file:
            raise ValueError(
                "Provide either swagger_url or swagger_file_path, not both"
            )
        if not has_url and not has_file:
            raise ValueError("Either swagger_url or swagger_file_path is required")
        return self
