"""Explicit cohort scope shared by dataset audit and materialization."""
from datafarm.common import read_json, sha256_file


def cohort_contract(args, samples):
    if args.training_scope not in {'production_policy', 'exploratory_diagnostic'}:
        raise ValueError('unknown training scope')
    count = args.expected_episodes
    if count < 1 or len(samples) != count:
        raise ValueError("source count differs from explicit expected episode count")
    ids = [row['sample_id'] for row in samples]
    if len(set(ids)) != count:
        raise ValueError("source identities must be unique")
    if args.training_scope == 'production_policy':
        return None
    if args.parent_execution_plan is None:
        raise ValueError("exploratory cohort requires a parent execution plan")
    parent = read_json(args.parent_execution_plan)
    parent_ids = [row['source_sample_id'] for row in parent['rows']]
    if len(set(parent_ids)) != len(parent_ids) or not set(ids) <= set(parent_ids):
        raise ValueError("selected identities differ from the fixed parent cohort")
    return {'parent_execution_plan': str(args.parent_execution_plan.resolve()),
            'parent_execution_plan_sha256': sha256_file(args.parent_execution_plan),
            'parent_count': len(parent_ids), 'selected_count': count,
            'selected_ids': sorted(ids), 'missing_ids': sorted(set(parent_ids) - set(ids)),
            'parent_complete': set(parent_ids) == set(ids)}

