"""Evidence-linked deterministic rule evaluation with explicit missing-data state.

No medication rules are bundled here. A caller must supply an externally
curated rule set and its evidence; this module only evaluates that rule set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RuleOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    CONTAINS = "contains"


class RuleEvaluationStatus(str, Enum):
    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class EvidenceReference:
    source_uri: str
    title: str
    version: str

    def __post_init__(self) -> None:
        for field in ("source_uri", "title", "version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Evidence {field} must be a non-empty string")


@dataclass(frozen=True)
class RuleCondition:
    fact_path: str
    operator: RuleOperator
    expected: Any

    def __post_init__(self) -> None:
        if not isinstance(self.fact_path, str) or not self.fact_path.strip():
            raise ValueError("fact_path must be non-empty")
        if any(not part for part in self.fact_path.split(".")):
            raise ValueError("fact_path cannot contain empty path segments")
        object.__setattr__(self, "operator", RuleOperator(self.operator))


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    version: str
    severity: str
    summary: str
    condition: RuleCondition
    evidence: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        for field in ("rule_id", "version", "severity", "summary"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if not self.evidence:
            raise ValueError("Rules without traceable evidence are rejected")


@dataclass(frozen=True)
class RuleFinding:
    rule_id: str
    version: str
    severity: str
    summary: str
    status: RuleEvaluationStatus
    fact_path: str
    observed_value: Any
    expected_value: Any
    evidence: tuple[EvidenceReference, ...]


def _lookup(facts: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = facts
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return current is not None, current


def _compare(actual: Any, condition: RuleCondition) -> bool | None:
    expected = condition.expected
    try:
        if condition.operator is RuleOperator.EQ:
            return actual == expected
        if condition.operator is RuleOperator.NE:
            return actual != expected
        if condition.operator is RuleOperator.GT:
            return actual > expected
        if condition.operator is RuleOperator.GTE:
            return actual >= expected
        if condition.operator is RuleOperator.LT:
            return actual < expected
        if condition.operator is RuleOperator.LTE:
            return actual <= expected
        if condition.operator is RuleOperator.IN:
            return actual in expected
        if condition.operator is RuleOperator.CONTAINS:
            return expected in actual
    except (TypeError, ValueError):
        return None
    return None


def evaluate_rules(
    rules: list[RuleDefinition] | tuple[RuleDefinition, ...],
    facts: Mapping[str, Any],
) -> tuple[RuleFinding, ...]:
    """Evaluate supplied rules deterministically; missing facts never mean safe."""
    if not isinstance(facts, Mapping):
        raise TypeError("facts must be a mapping")
    ordered_rules = sorted(rules, key=lambda rule: (rule.rule_id, rule.version))
    identifiers = [rule.rule_id for rule in ordered_rules]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("rule_id values must be unique in one evaluation")

    findings: list[RuleFinding] = []
    for rule in ordered_rules:
        found, actual = _lookup(facts, rule.condition.fact_path)
        if found and isinstance(actual, (int, float)) and not isinstance(actual, bool):
            found = math.isfinite(actual)
        matched = _compare(actual, rule.condition) if found else None
        if matched is None:
            status = RuleEvaluationStatus.INSUFFICIENT_DATA
        elif matched:
            status = RuleEvaluationStatus.TRIGGERED
        else:
            status = RuleEvaluationStatus.NOT_TRIGGERED
        findings.append(
            RuleFinding(
                rule_id=rule.rule_id,
                version=rule.version,
                severity=rule.severity,
                summary=rule.summary,
                status=status,
                fact_path=rule.condition.fact_path,
                observed_value=actual if found else None,
                expected_value=rule.condition.expected,
                evidence=rule.evidence,
            )
        )
    return tuple(findings)
