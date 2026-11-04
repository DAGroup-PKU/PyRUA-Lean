"""Every physical quantity in G says how it is known; uncertain ones are randomized."""
import copy
import json

import pytest

from helpers import NOW, payload_for
from datafarm.core.grounding import GroundingError, validate_grounding, validate_quantities
from datafarm.core.model import ArtifactDocument, IRKind
from datafarm.core.schema import SchemaError, validate_artifact
from datafarm.task.contract import verify_delivery
from test_white_delivery import fixture

EVIDENCE = {'path': 'calibration/scale.json', 'sha256': '0' * 64}
LEDGER = [
    {'name': 'table_height', 'label': 'fitted', 'unit': 'm', 'value': 0.752, 'evidence': [EVIDENCE],
     'randomization': {'low': 0.732, 'high': 0.772}},
    {'name': 'gripper_width', 'label': 'measured', 'unit': 'm', 'value': 0.08, 'evidence': [EVIDENCE]},
    {'name': 'cup_mass', 'label': 'assumed', 'unit': 'kg', 'value': 0.02,
     'randomization': {'low': 0.01, 'high': 0.04}},
    {'name': 'cloth_bending_stiffness', 'label': 'unknown', 'unit': '', 'value': None,
     'randomization': {'choices': [0.1, 0.3, 1.0]}},
    {'name': 'joint_friction', 'label': 'unknown', 'unit': 'N*m', 'value': None,
     'not_randomized_reason': 'position-controlled joints; friction does not enter the rollout'},
]


def test_complete_ledger_counts_labels():
    assert validate_quantities(LEDGER) == {'measured': 1, 'fitted': 1, 'assumed': 1, 'unknown': 2}


@pytest.mark.parametrize('index, change, message', [
    (0, {'evidence': []}, 'needs evidence'),
    (1, {'label': 'guessed'}, 'label must be one of'),
    (2, {'randomization': None}, 'randomization must be an object'),
    (2, {'randomization': {'low': 0.03, 'high': 0.04}}, 'must contain the nominal value'),
    (2, {'value': None}, 'only unknown quantities may omit the value'),
    (3, {'randomization': {'choices': [0.1]}}, 'at least two values'),
    (4, {'not_randomized_reason': ' '}, 'needs randomization or not_randomized_reason'),
    (1, {'value': float('nan')}, 'finite number'),
    (1, {'name': 'table_height'}, 'duplicate physical quantity'),
])
def test_unsupported_or_unrandomized_values_are_refused(index, change, message):
    ledger = copy.deepcopy(LEDGER)
    ledger[index].update(change)
    if change.get('randomization', 0) is None:
        ledger[index].pop('randomization')
        ledger[index]['randomization'] = None
    with pytest.raises(GroundingError, match=message):
        validate_quantities(ledger)


def test_assumed_value_without_randomization_or_reason_is_refused():
    ledger = copy.deepcopy(LEDGER)
    ledger[2].pop('randomization')
    with pytest.raises(GroundingError, match='needs randomization or not_randomized_reason'):
        validate_quantities(ledger)


def test_record_needs_schema_and_evidence_binding_is_applied():
    with pytest.raises(GroundingError, match='schema'):
        validate_grounding({'physical_quantities': LEDGER})
    seen = []
    validate_grounding({'schema': 'datafarm.grounding/1', 'physical_quantities': LEDGER}, bind=seen.append)
    assert seen == [EVIDENCE, EVIDENCE]


def grounding_ir(identification):
    payload = payload_for(IRKind.GROUNDING)
    payload['system_identification'] = identification
    return ArtifactDocument(kind=IRKind.GROUNDING, artifact_id='fixture:grounding', created_at=NOW,
                            producer='grounding-compiler', predecessors=(), evidence=(), payload=payload)


def test_grounding_ir_payload_requires_a_typed_ledger():
    ir_evidence = {'kind': 'camera-fit', 'sha256': '0' * 64, 'media_type': 'application/json'}
    ledger = [{**entry, 'evidence': [ir_evidence]} if entry.get('evidence') else entry for entry in LEDGER]
    validate_artifact(grounding_ir({'physical_quantities': ledger}))
    with pytest.raises(SchemaError, match='must contain exactly'):
        validate_artifact(grounding_ir({'physical_quantities': [{**LEDGER[1], 'evidence': [{'sha256': '0' * 64}]}]}))
    with pytest.raises(SchemaError, match='physical_quantities'):
        validate_artifact(grounding_ir({}))
    with pytest.raises(SchemaError, match='randomization'):
        validate_artifact(grounding_ir({'physical_quantities': [
            {'name': 'mass', 'label': 'assumed', 'unit': 'kg', 'value': 0.2}]}))


@pytest.mark.parametrize('mutation', ['missing', 'unlabelled', 'changed-evidence'])
def test_delivery_verifies_the_bound_grounding_record(tmp_path, mutation):
    d, _, put = fixture(tmp_path)
    assert verify_delivery(tmp_path)['physical_quantities']['fitted'] == 1
    if mutation == 'missing':
        d.pop('grounding')
    elif mutation == 'unlabelled':
        d['grounding'] = put('grounding.json', {'schema': 'datafarm.grounding/1', 'physical_quantities': [
            {'name': 'table_height', 'unit': 'm', 'value': 0.75}]})
    else:
        (tmp_path / 'input-source.json').write_text(json.dumps({'synthetic': 'changed'}))
    put('delivery.json', d)
    with pytest.raises(ValueError):
        verify_delivery(tmp_path)
