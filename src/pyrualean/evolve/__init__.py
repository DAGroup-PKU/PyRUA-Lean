# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Iterating a skill library on dev layouts: what the runs say, and whether a library is better.

* :mod:`.digest` condenses one episode's run directory into what a reviewer reads first: the
  verdict and how far it got, the cells, and the primitive calls that failed.
* :mod:`.gate` compares two libraries on the same layouts: success, then stages passed, then
  score; a tie keeps the parent.

The serial online entry point is :mod:`.online`: it composes these utilities with a
fresh-process launcher, dev-only reflection, budgets and paired validation. Its configuration
defaults to launch disabled. The library, digest and gate remain usable without a simulator.
"""
