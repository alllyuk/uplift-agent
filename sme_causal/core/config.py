"""
Centralized configuration management for SME causal graph inference project.

This module provides a unified configuration system using Pydantic settings
for all application parameters, environment variables, and defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    """Configuration for file paths and directories."""

    # Base directories
    # Ensure project_root resolves to repository root regardless of package nesting
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )
    artifacts_dir: Path = Field(
        default_factory=lambda: (Path(__file__).resolve().parents[2] / "artifacts")
    )

    rag_data_dir: Path = Field(
        default_factory=lambda: (Path(__file__).resolve().parents[2] / "rag_data")
    )

    document_corpus_dir: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2] / "rag_data" / "document_corpus"
        )
    )

    raw_documents_dir: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2] / "rag_data" / "raw_documents"
        )
    )

    cleaned_documents_dir: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2] / "rag_data" / "cleaned_documents"
        )
    )

    algorithmic_dir: Path = Field(
        default_factory=lambda: (Path(__file__).resolve().parents[2] / "causal_outputs")
    )

    # Data files
    synthetic_clients_csv: str = "synthetic_clients.csv"
    ground_truth_edges_json: str = "ground_truth_edges.json"
    llm_edges_json: str = "llm_edges.json"
    algo_edges_json: str = "graph_consensus.json"
    edge_report_csv: str = "edge_report.csv"
    pipeline_log: str = "pipeline.log"
    streamlit_log: str = "streamlit.log"
    metadata_csv: str = "metadata.csv"
    chunks_parquet: str = "chunks.parquet"
    embeddings_parquet: str = "embeddings.parquet"
    index_faiss: str = "index.faiss"

    # Graph files
    graph_prefix: str = "graph_merged"

    @field_validator("artifacts_dir", mode="before")
    @classmethod
    def ensure_artifacts_dir_exists(cls, path: Path | str) -> Path:
        """Ensure artifacts directory exists.

        Accepts either a Path or a string (e.g., from env var) and ensures the
        directory is created.
        """
        p = path if isinstance(path, Path) else Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = SettingsConfigDict(
        env_prefix="PATHS_", env_file=".env", extra="ignore"
    )


class DataGenerationConfig(BaseSettings):
    """Configuration for synthetic data generation."""

    n_clients: int = Field(
        default=3000,
        ge=1,
        le=100000,
        description="Number of synthetic clients to generate",
    )
    seed: int = Field(
        default=42, ge=0, description="Random seed for reproducible generation"
    )

    model_config = SettingsConfigDict(
        env_prefix="DATA_", env_file=".env", extra="ignore"
    )


class LLMConfig(BaseSettings):
    """Configuration for LLM-based causal inference."""

    # Model settings
    provider: Literal["openai", "local"] = Field(
        default="openai", description="LLM provider"
    )

    model_name: str = Field(
        default="gpt-5.4-mini",
        validation_alias="LLM_MODEL",
        description="OpenAI model name",
    )
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Sampling temperature"
    )
    bootstrap_rounds: int = Field(
        default=2, ge=1, le=10, description="Number of bootstrap rounds"
    )
    sample_rows: int = Field(
        default=900, ge=10, le=10000, description="Maximum rows to sample"
    )

    # Inference parameters
    confidence_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    votes_threshold: int = Field(default=1, ge=1, description="Minimum votes threshold")

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", extra="ignore"
    )


class APIConfig(BaseSettings):
    """Configuration for external API services."""

    openai_api_key: Optional[str] = Field(
        None, description="OpenAI API key"
    )
    llm_base_url: Optional[str] = Field(
        None, description="OpenAI-compatible API base URL (optional)"
    )

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")


class LoggingConfig(BaseSettings):
    """Configuration for logging system."""

    level: str = Field(default="INFO", description="Logging level")
    rotation: str = Field(default="10 MB", description="Log file rotation size")
    retention: str = Field(default="1 week", description="Log retention period")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log message format",
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_", env_file=".env", extra="ignore"
    )


class StreamlitConfig(BaseSettings):
    """Configuration for Streamlit web interface."""

    page_title: str = "SME Causal Graph Demo"
    page_icon: str = "📈"
    layout: str = "wide"
    theme_primary_color: Optional[str] = None
    theme_background_color: Optional[str] = None
    theme_secondary_background_color: Optional[str] = None
    theme_text_color: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="STREAMLIT_", env_file=".env", extra="ignore"
    )


class HybridGraphConfig(BaseSettings):
    """Configuration for building a hybrid graph"""

    confidence_threshold: float = Field(
        default=0.6, description="Minimum confidence for edge inclusion"
    )


class AppConfig(BaseSettings):
    """Main application configuration combining all sub-configs."""

    # Sub-configurations
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data_generation: DataGenerationConfig = Field(default_factory=DataGenerationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    hybrid_graph: HybridGraphConfig = Field(default_factory=HybridGraphConfig)

    # Application metadata
    version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")

    @property
    def full_artifacts_dir(self) -> Path:
        """Get the full path to artifacts directory."""
        # artifacts_dir is already absolute by default; return directly
        return self.paths.artifacts_dir

    @property
    def rag_data_dir(self) -> Path:
        """Get the full path to rag artifacts directory."""
        # artifacts_dir is already absolute by default; return directly
        return self.paths.rag_data_dir

    @property
    def document_corpus_dir(self) -> Path:
        """Get the full path to document corpus directory."""
        # artifacts_dir is already absolute by default; return directly
        return self.paths.document_corpus_dir
    
    @property
    def raw_documents_dir(self) -> Path:
        """Get the full path to raw pdf documents directory."""
        # artifacts_dir is already absolute by default; return directly
        return self.paths.raw_documents_dir
    
    @property
    def cleaned_documents_dir(self) -> Path:
        """Get the full path to raw pdf documents directory."""
        # artifacts_dir is already absolute by default; return directly
        return self.paths.cleaned_documents_dir

    @property
    def full_algorithmic_dir(self) -> Path:
        """Get the full path to algo-artifacts directory."""
        # artifacts_dir is already absolute by default; return directly
        return self.paths.algorithmic_dir

    @property
    def synthetic_clients_path(self) -> Path:
        """Get full path to synthetic clients CSV file."""
        return self.full_artifacts_dir / self.paths.synthetic_clients_csv

    @property
    def chunks_path(self) -> Path:
        """Get full path to chunks.parquet"""
        return self.rag_data_dir / self.paths.chunks_parquet

    @property
    def documents_metadata_path(self) -> Path:
        """Get full path to documents metadata.csv for rag"""
        return self.document_corpus_dir / self.paths.metadata_csv

    @property
    def embeddings_path(self) -> Path:
        """Get full path to embeddings.parquet"""
        return self.rag_data_dir / self.paths.embeddings_parquet

    @property
    def faiss_index_path(self) -> Path:
        """Get full path to faiss index file"""
        return self.rag_data_dir / self.paths.index_faiss

    @property
    def ground_truth_edges_path(self) -> Path:
        """Get full path to ground truth edges JSON file."""
        return self.full_artifacts_dir / self.paths.ground_truth_edges_json

    @property
    def llm_edges_path(self) -> Path:
        """Get full path to LLM edges JSON file."""
        return self.full_artifacts_dir / self.paths.llm_edges_json

    @property
    def algo_edges_path(self) -> Path:
        """Get full path to Algorithmic edges JSON file."""
        return self.full_algorithmic_dir / self.paths.algo_edges_json

    @property
    def edge_report_path(self) -> Path:
        """Get full path to edge report CSV file."""
        return self.full_artifacts_dir / self.paths.edge_report_csv

    @property
    def pipeline_log_path(self) -> Path:
        """Get full path to pipeline log file."""
        return self.full_artifacts_dir / self.paths.pipeline_log

    @property
    def streamlit_log_path(self) -> Path:
        """Get full path to Streamlit log file."""
        return self.full_artifacts_dir / self.paths.streamlit_log

    @property
    def cases_db_path(self) -> Path:
        """Get full path to SQLite cases database."""
        return self.full_artifacts_dir / "cases.db"

    @property
    def effective_llm_api_key(self) -> Optional[str]:
        """Return effective OpenAI API key from env or config.

        Uses the standard `OPENAI_API_KEY` environment variable.
        """
        import os

        return self._first_non_empty(
            os.getenv("OPENAI_API_KEY"),
            self.api.openai_api_key,
        )

    @property
    def effective_openai_api_key(self) -> Optional[str]:
        """Return effective OpenAI API key from env or config.

        Uses the same API key resolution as `effective_llm_api_key`.
        """
        return self.effective_llm_api_key

    @staticmethod
    def _first_non_empty(*values: Optional[str]) -> Optional[str]:
        for value in values:
            if value is None:
                continue
            normalized = str(value).strip()
            if normalized:
                return normalized
        return None

    @property
    def effective_llm_provider(self) -> str:
        """
        берём provider из LLMConfig (он подхватывает LLM_PROVIDER из .env)
        как fallback можно глянуть os.getenv (на случай, если переменная задана в ОС)
        """
        import os

        # Pydantic уже учёл .env для self.llm.provider
        provider = (self.llm.provider or "openai").lower()

        # если переменная реально задана в окружении процесса — переопределим
        env_override = os.getenv("LLM_PROVIDER")
        if env_override:
            provider = env_override.lower()

        return provider

    @property
    def effective_llm_base_url(self) -> Optional[str]:
        """Return effective OpenAI base URL from env or config.

        Prefers the environment variable `OPENAI_API_BASE` when set; otherwise
        falls back to the value from .env via api.openai_base_url.
        """
        import os

        if self.effective_llm_provider == "local":
            return os.getenv("LLM_BASE_URL") or self.api.llm_base_url
        return None  # Not needed for 'openai' provider

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Global configuration instance and accessor
config = AppConfig()


def get_config() -> AppConfig:
    """Return the global application configuration instance."""
    return config
