"""Typed error hierarchy with stable, unique error codes.

Author: 晨星
"""

from __future__ import annotations

from typing import ClassVar


class OperatorForgeError(Exception):
    """Base exception carrying a stable machine-readable code."""

    code: ClassVar[str] = "E000"

    def __init__(self, message: str) -> None:
        if not message.strip():
            message = "OperatorForge encountered an unspecified error."
        super().__init__(f"{self.code}: {message}")
        self.message = message


class DataError(OperatorForgeError):
    """Raised when a dataset or PDE input is invalid."""

    code = "E100"


class ModelError(OperatorForgeError):
    """Raised when an operator cannot be constructed or evaluated."""

    code = "E200"


class TrainingError(OperatorForgeError):
    """Raised when optimization cannot proceed safely."""

    code = "E300"


class EvaluationError(OperatorForgeError):
    """Raised when a metric receives incompatible arrays."""

    code = "E400"


class ConfigurationError(OperatorForgeError):
    """Raised when configuration values fail schema validation."""

    code = "E500"


def validate_error_registry() -> None:
    """Fail fast if an error code or class name is duplicated.

    Python silently shadows duplicate class declarations in a module.  Walking
    subclasses additionally catches accidental duplicates imported by plugins.
    """
    seen_codes: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    pending: list[type[OperatorForgeError]] = list(OperatorForgeError.__subclasses__())
    while pending:
        error_type = pending.pop()
        pending.extend(error_type.__subclasses__())
        name = error_type.__name__
        code = error_type.code
        if code in seen_codes:
            raise RuntimeError(f"Duplicate error code {code}: {seen_codes[code]} and {name}")
        if name in seen_names:
            raise RuntimeError(f"Duplicate error class name {name}")
        seen_codes[code] = name
        seen_names[name] = code


validate_error_registry()

__all__ = [
    "ConfigurationError",
    "DataError",
    "EvaluationError",
    "ModelError",
    "OperatorForgeError",
    "TrainingError",
    "validate_error_registry",
]
