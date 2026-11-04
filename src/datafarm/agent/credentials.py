"""Credential providers for model clients.

Clients never read environment variables or well-known paths for keys. The caller
injects a provider; the standard one parses ``KEY=VALUE`` lines (optionally prefixed
with ``export``) from a keys file whose path the caller passes explicitly. Values never
appear in ``repr``, exceptions, receipts or logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Callable


class CredentialError(RuntimeError):
    """A credential could not be loaded. Messages never contain secret values."""


@dataclass(frozen=True, repr=False)
class Credentials:
    api_key: str
    endpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key or any(ch.isspace() for ch in self.api_key):
            raise CredentialError("API key is empty or contains whitespace")

    def __repr__(self) -> str:
        return f"Credentials(api_key=<redacted>, endpoint={self.endpoint!r})"


CredentialProvider = Callable[[], Credentials]


def parse_keys_text(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; ``#`` comments, blank lines and ``export`` are allowed."""

    values: dict[str, str] = {}
    for number, original in enumerate(text.splitlines(), start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        if not key.replace("_", "").isalnum():
            raise CredentialError(f"keys file line {number}: invalid variable name")
        try:
            parsed = shlex.split(raw.strip())
        except ValueError:
            raise CredentialError(f"keys file line {number}: unbalanced quoting") from None
        values[key] = parsed[0] if parsed else ""
    return values


class KeysFileCredentials:
    """Load an API key (and optional endpoint) from a caller-supplied keys file.

    The file is read lazily on first use and cached in memory only.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        key_var: str = "API_KEY",
        endpoint_var: str | None = "END_POINT",
    ) -> None:
        self.path = Path(path)
        self.key_var = key_var
        self.endpoint_var = endpoint_var
        self._cached: Credentials | None = None

    def __repr__(self) -> str:
        return f"KeysFileCredentials(path={str(self.path)!r}, key_var={self.key_var!r})"

    def __call__(self) -> Credentials:
        if self._cached is None:
            try:
                text = self.path.read_text(encoding="utf-8")
            except OSError as error:
                raise CredentialError(
                    f"cannot read keys file {self.path}: {type(error).__name__}"
                ) from None
            values = parse_keys_text(text)
            key = values.get(self.key_var)
            if not key:
                raise CredentialError(f"{self.key_var} is missing from {self.path}")
            endpoint = values.get(self.endpoint_var) if self.endpoint_var else None
            self._cached = Credentials(api_key=key, endpoint=endpoint or None)
        return self._cached

