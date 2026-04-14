"""Unit tests for the prompt version registry."""

from __future__ import annotations

import pytest

from sme_causal.agent.prompt_registry import (
    PromptVersionError,
    ensure_versions,
    list_versions,
    load_metadata,
    version_exists,
)


class TestRegistry:
    def test_base_v10_registered(self):
        assert version_exists("base", "v1.0")

    def test_whatif_v10_registered(self):
        assert version_exists("whatif", "v1.0")

    def test_missing_version(self):
        assert not version_exists("base", "v9.9")

    def test_list_versions_base(self):
        versions = list_versions("base")
        assert "v1.0" in versions

    def test_list_versions_unknown(self):
        assert list_versions("doesnotexist") == []

    def test_load_metadata(self):
        meta = load_metadata("whatif", "v1.0")
        assert meta.get("version") == "v1.0"
        assert "variables" in meta

    def test_load_metadata_missing_raises(self):
        with pytest.raises(PromptVersionError):
            load_metadata("whatif", "v99.99")

    def test_ensure_versions_ok(self):
        ensure_versions({"base": "v1.0", "whatif": "v1.0"})

    def test_ensure_versions_missing(self):
        with pytest.raises(PromptVersionError) as exc_info:
            ensure_versions({"base": "v1.0", "whatif": "v9.9"})
        assert "whatif:v9.9" in str(exc_info.value)
