from __future__ import annotations

import unittest

from datafarm.core.model import ArtifactDocument, IRKind
from datafarm.core.schema import SchemaError, validate_artifact

from tests.helpers import NOW, payload_for


class SchemaTests(unittest.TestCase):
    def test_all_current_ir_types_validate(self) -> None:
        for kind in IRKind:
            with self.subTest(kind=kind.value):
                document = ArtifactDocument(
                    kind=kind,
                    artifact_id=f"fixture:{kind.value}",
                    created_at=NOW,
                    producer="fixture-compiler",
                    predecessors=(),
                    evidence=(),
                    payload=payload_for(kind),
                )
                validate_artifact(document)

    def test_missing_required_section_is_rejected(self) -> None:
        payload = payload_for(IRKind.EPISODE)
        payload.pop("frame_map")
        document = ArtifactDocument(
            kind=IRKind.EPISODE,
            artifact_id="fixture:episode",
            created_at=NOW,
            producer="episode-compiler",
            predecessors=(),
            evidence=(),
            payload=payload,
        )
        with self.assertRaisesRegex(SchemaError, "frame_map"):
            validate_artifact(document)

    def test_absolute_runtime_path_is_rejected(self) -> None:
        payload = payload_for(IRKind.GROUNDING)
        payload["source"] = {"location": "/some/research/workspace/input.mp4"}
        document = ArtifactDocument(
            kind=IRKind.GROUNDING,
            artifact_id="fixture:grounding",
            created_at=NOW,
            producer="grounding-compiler",
            predecessors=(),
            evidence=(),
            payload=payload,
        )
        with self.assertRaisesRegex(SchemaError, "absolute filesystem path"):
            validate_artifact(document)

    def test_secret_field_is_rejected(self) -> None:
        payload = payload_for(IRKind.RENDER)
        payload["backend"] = {"api_key": "not-allowed"}
        document = ArtifactDocument(
            kind=IRKind.RENDER,
            artifact_id="fixture:render",
            created_at=NOW,
            producer="render-compiler",
            predecessors=(),
            evidence=(),
            payload=payload,
        )
        with self.assertRaisesRegex(SchemaError, "secret field"):
            validate_artifact(document)


if __name__ == "__main__":
    unittest.main()
