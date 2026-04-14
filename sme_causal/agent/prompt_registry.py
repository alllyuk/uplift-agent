"""Prompt version registry.

PoC-подход: реестр `prompts/{name}/{version}.yaml` содержит метаданные версии
(version, created_at, parent_version, notes, variables). Фактический текст
шаблонов остаётся inline в `CausalAgent` — это осознанный компромисс: runtime
использует inline-строки (стабильность), а YAML-реестр обеспечивает трекинг
активных версий для `CaseState.prompt_versions` и SQLite audit.

Полный перенос текста в YAML с runtime-загрузкой — задача для v2, когда
появится >1 версии и потребуется переключение через config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml


# Корень реестра — каталог `prompts/` в корне репозитория
_REGISTRY_ROOT = Path(__file__).resolve().parents[2] / "prompts"


class PromptVersionError(Exception):
    """Запрошенной версии промпта нет в реестре."""


def _version_path(name: str, version: str) -> Path:
    return _REGISTRY_ROOT / name / f"{version}.yaml"


def version_exists(name: str, version: str) -> bool:
    return _version_path(name, version).is_file()


def load_metadata(name: str, version: str) -> Dict[str, object]:
    """Прочитать метаданные версии промпта.

    Бросает PromptVersionError, если версия не зарегистрирована.
    """
    path = _version_path(name, version)
    if not path.is_file():
        raise PromptVersionError(
            f"Prompt version not found: name={name!r}, version={version!r}, "
            f"expected at {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise PromptVersionError(
            f"Malformed prompt registry file {path}: expected dict at top level"
        )
    return data


def list_versions(name: str) -> List[str]:
    """Список доступных версий промпта по имени (отсортированный)."""
    name_dir = _REGISTRY_ROOT / name
    if not name_dir.is_dir():
        return []
    return sorted(p.stem for p in name_dir.glob("*.yaml"))


def ensure_versions(versions: Dict[str, str]) -> None:
    """Проверить, что все пары name -> version зарегистрированы.

    Используется при старте Pipeline/агента, чтобы рано словить typo в config.
    """
    missing = [
        (name, ver)
        for name, ver in versions.items()
        if not version_exists(name, ver)
    ]
    if missing:
        details = ", ".join(f"{n}:{v}" for n, v in missing)
        raise PromptVersionError(
            f"Unknown prompt versions in registry: {details}"
        )
