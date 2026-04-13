"""Shared type definitions for the project."""

from __future__ import annotations

from typing import Literal, TypedDict, List, Optional


Polarity = Literal["+", "-", "non-monotonic"]


class Edge(TypedDict, total=False):
    source: str
    target: str
    relation: str
    polarity: Polarity
    confidence: float
    rationale: str


class ExplanationDict(TypedDict, total=False):
    diagnosis: str
    drivers_pos: List[str]
    drivers_neg: List[str]
    recommendations: List[str]
    expected_effect: str
    raw_text: str

