"""Physical-quantity ledger of the grounding record G.

A calibration video determines some quantities and leaves others unidentifiable. Every
physical quantity the simulator uses is therefore listed with how it is known:

- measured: read from a sensor, datasheet or the robot description; needs evidence.
- fitted: estimated from the real episode (camera fit, scale, table height); needs evidence.
- assumed: a chosen value without task evidence (e.g. a friction prior).
- unknown: not determined; the value may be null.

An assumed or unknown quantity must be covered by randomization (a range or a set of
choices) or carry an explicit reason why it is not randomized, so a confident but
unsupported value cannot enter the simulator silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from datafarm.common import is_finite_number

GROUNDING_SCHEMA = 'datafarm.grounding/1'
QUANTITY_LABELS = ('measured', 'fitted', 'assumed', 'unknown')
EVIDENCE_LABELS = {'measured', 'fitted'}
UNCERTAIN_LABELS = {'assumed', 'unknown'}


class GroundingError(ValueError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise GroundingError(message)


def _randomization(name: str, value: Any, spec: Any) -> None:
    _check(isinstance(spec, dict), f'{name}: randomization must be an object')
    if 'choices' in spec:
        _check(isinstance(spec['choices'], list) and len(spec['choices']) >= 2,
               f'{name}: randomization choices need at least two values')
        return
    low, high = spec.get('low'), spec.get('high')
    _check(is_finite_number(low) and is_finite_number(high) and low < high, f'{name}: randomization needs low < high')
    if is_finite_number(value):
        _check(low <= value <= high, f'{name}: randomization range must contain the nominal value')


def validate_quantities(entries: Any, *, bind: Callable[[dict], Any] | None = None) -> dict[str, int]:
    """Validate a physical_quantities list.

    `bind` verifies each evidence reference (a bound workspace file for deliveries, a
    content-addressed ArtifactRef in the IR); without it a path/sha256 shape is required.

    Returns the number of quantities per label.
    """
    _check(isinstance(entries, list), 'physical_quantities must be a list')
    counts = {label: 0 for label in QUANTITY_LABELS}
    names = set()
    for index, entry in enumerate(entries):
        _check(isinstance(entry, dict), f'physical_quantities[{index}] must be an object')
        name = entry.get('name')
        _check(isinstance(name, str) and bool(name.strip()), f'physical_quantities[{index}] needs a name')
        _check(name not in names, f'duplicate physical quantity: {name}')
        names.add(name)
        label = entry.get('label')
        _check(label in QUANTITY_LABELS, f'{name}: label must be one of {list(QUANTITY_LABELS)}')
        _check(isinstance(entry.get('unit'), str), f'{name}: unit must be a string ("" if dimensionless)')
        value = entry.get('value')
        _check(value is None or is_finite_number(value) or isinstance(value, (list, str)),
               f'{name}: value must be a finite number, list, string or null')
        _check(value is not None or label == 'unknown', f'{name}: only unknown quantities may omit the value')
        evidence = entry.get('evidence', [])
        _check(isinstance(evidence, list), f'{name}: evidence must be a list')
        if label in EVIDENCE_LABELS:
            _check(bool(evidence), f'{name}: {label} quantity needs evidence')
        for ref in evidence:
            _check(isinstance(ref, dict), f'{name}: evidence entries must be objects')
            if bind is None:
                _check(isinstance(ref.get('path'), str) and isinstance(ref.get('sha256'), str),
                       f'{name}: evidence must be path/sha256 references')
            else:
                bind(ref)
        if label in UNCERTAIN_LABELS:
            reason = entry.get('not_randomized_reason')
            if 'randomization' in entry:
                _randomization(name, value, entry['randomization'])
            else:
                _check(isinstance(reason, str) and bool(reason.strip()),
                       f'{name}: {label} quantity needs randomization or not_randomized_reason')
        elif 'randomization' in entry:
            _randomization(name, value, entry['randomization'])
        counts[label] += 1
    return counts


def validate_grounding(document: Any, *, bind: Callable[[dict], Any] | None = None) -> dict[str, int]:
    _check(isinstance(document, dict) and document.get('schema') == GROUNDING_SCHEMA,
           f'grounding record must use schema {GROUNDING_SCHEMA}')
    return validate_quantities(document.get('physical_quantities'), bind=bind)


def load_grounding(path: Path, *, bind: Callable[[dict], Any] | None = None) -> dict[str, int]:
    import json
    return validate_grounding(json.loads(Path(path).read_text()), bind=bind)
