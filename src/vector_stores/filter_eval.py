"""Filter evaluation for OpenAI-compatible ComparisonFilter and CompoundFilter.

Given a Pydantic-validated filter object (constructed from the API request) and
a chunk's attributes dict, returns whether the chunk matches.
"""

from __future__ import annotations

from typing import Any, Callable

from apps.api.schemas.vector_stores import (
    AndFilter,
    ComparisonFilter,
    CompoundFilter,
    OrFilter,
)


def _eval_comparison(cf: ComparisonFilter, attrs: dict[str, Any]) -> bool:
    if cf.key not in attrs:
        return False
    actual = attrs[cf.key]
    expected = cf.value
    op = cf.type
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "gt":
        try:
            return actual > expected
        except TypeError:
            return False
    if op == "gte":
        try:
            return actual >= expected
        except TypeError:
            return False
    if op == "lt":
        try:
            return actual < expected
        except TypeError:
            return False
    if op == "lte":
        try:
            return actual <= expected
        except TypeError:
            return False
    if op == "in":
        if not isinstance(expected, list):
            return False
        return actual in expected
    if op == "nin":
        if not isinstance(expected, list):
            return False
        return actual not in expected
    return False


def evaluate(f: CompoundFilter | None, attrs: dict[str, Any]) -> bool:
    """Evaluate an OpenAI compound filter against a chunk's attributes.

    A ``None`` filter matches everything.
    """
    if f is None:
        return True
    if isinstance(f, ComparisonFilter):
        return _eval_comparison(f, attrs)
    if isinstance(f, AndFilter):
        return all(evaluate(sub, attrs) for sub in f.filters)
    if isinstance(f, OrFilter):
        return any(evaluate(sub, attrs) for sub in f.filters)
    return False


def compile_predicate(
    f: CompoundFilter | None,
) -> Callable[[dict[str, Any]], bool]:
    """Return a predicate function bound to the given filter."""

    def predicate(attrs: dict[str, Any]) -> bool:
        return evaluate(f, attrs)

    return predicate
