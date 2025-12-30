from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Type
from pathlib import Path


class SwaggerProcessingRequest(BaseModel):
    """
    Request model for processing Swagger/OpenAPI specifications.

    Exactly one of swagger_url or swagger_path must be provided.
    - swagger_url: HTTP(S) URL to fetch the schema from
    - swagger_path: Local filesystem path to read the schema from
    """
    swagger_url: Optional[str] = None
    swagger_path: Optional[str] = None

    @field_validator("swagger_url", mode="before")
    @classmethod
    def coerce_url_to_string(
        cls: Type["SwaggerProcessingRequest"], v: Optional[str]
    ) -> Optional[str]:
        if v is None:
            return v
        return str(v).strip()

    @field_validator("swagger_path", mode="before")
    @classmethod
    def coerce_path_to_string(
        cls: Type["SwaggerProcessingRequest"], v: Optional[str]
    ) -> Optional[str]:
        if v is None:
            return v
        # Convert Path objects to string
        if isinstance(v, Path):
            return str(v)
        return str(v).strip()

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> "SwaggerProcessingRequest":
        """Ensure exactly one of swagger_url or swagger_path is provided"""
        has_url = self.swagger_url is not None and self.swagger_url.strip() != ""
        has_path = self.swagger_path is not None and self.swagger_path.strip() != ""

        if has_url and has_path:
            raise ValueError(
                "Cannot specify both swagger_url and swagger_path. "
                "Please provide only one source."
            )

        if not has_url and not has_path:
            raise ValueError(
                "Must specify either swagger_url or swagger_path. "
                "No source provided."
            )

        return self

    @property
    def is_url_source(self) -> bool:
        """Check if the source is a URL"""
        return self.swagger_url is not None and self.swagger_url.strip() != ""

    @property
    def is_file_source(self) -> bool:
        """Check if the source is a file path"""
        return self.swagger_path is not None and self.swagger_path.strip() != ""

    @property
    def source_location(self) -> str:
        """Get the source location (URL or path)"""
        if self.is_url_source:
            return self.swagger_url
        return self.swagger_path
