# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Settings read from the environment: ``PYRUALEAN_<NAME>``.

Every environment variable of this package is named ``PYRUALEAN_<NAME>``.
"""

from __future__ import annotations

import os

#: Prefix of every environment variable the package reads.
PREFIX = "PYRUALEAN_"


def setting_var(name: str) -> str:
    """The variable :func:`setting` reads for ``name``."""

    return PREFIX + name


def setting(name: str, default: str | None = None) -> str | None:
    """``$PYRUALEAN_<name>``, else ``default``.

    As with ``os.environ.get``, a variable set to the empty string counts as set.
    """

    return os.environ.get(setting_var(name), default)


__all__ = ["PREFIX", "setting", "setting_var"]
