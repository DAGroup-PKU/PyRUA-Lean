from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from datafarm.core.model import ArtifactDocument, GateReceipt
from datafarm.core.schema import canonical_json_bytes, validate_artifact, validate_gate_receipt
from datafarm.doctor import add_parsers as add_doctor_parser, handle as handle_doctor
from datafarm.common import sha256_bytes


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate(path: Path) -> int:
    value = _load(path)
    schema = value.get("schema")
    if schema == "datafarm.gate-receipt":
        validate_gate_receipt(GateReceipt.from_dict(value))
    elif isinstance(schema, str) and schema.startswith("datafarm."):
        validate_artifact(ArtifactDocument.from_dict(value))
    else:
        raise ValueError(f"unsupported schema {schema!r}")
    digest = sha256_bytes(canonical_json_bytes(value))
    print(json.dumps({"valid": True, "sha256": digest, "schema": schema}, indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datafarm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a current artifact or receipt")
    validate.add_argument("path", type=Path)
    add_doctor_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "validate":
        return _validate(args.path)
    if args.command == "doctor":
        return handle_doctor(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
