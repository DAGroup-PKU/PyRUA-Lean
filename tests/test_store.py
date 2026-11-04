from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datafarm.core.model import ArtifactDocument, IRKind
from datafarm.core.store import ArtifactIntegrityError, ArtifactStore

from tests.helpers import NOW, payload_for


class StoreTests(unittest.TestCase):
    def _document(self) -> ArtifactDocument:
        return ArtifactDocument(
            kind=IRKind.GROUNDING,
            artifact_id="fixture:grounding",
            created_at=NOW,
            producer="grounding-compiler",
            predecessors=(),
            evidence=(),
            payload=payload_for(IRKind.GROUNDING),
        )

    def test_same_document_has_same_content_address(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            first = store.put_document(self._document())
            second = store.put_document(self._document())
            self.assertEqual(first, second)
            self.assertEqual(store.document(first), self._document())

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            reference = store.put_document(self._document())
            stored = Path(directory) / "sha256" / reference.sha256[:2] / f"{reference.sha256}.json"
            stored.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactIntegrityError, "digest mismatch"):
                store.load(reference)

    def test_binary_evidence_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            reference = store.put_bytes("source-video", b"video-bytes", media_type="video/mp4")
            self.assertEqual(store.load_bytes(reference), b"video-bytes")
            with self.assertRaisesRegex(ArtifactIntegrityError, "use load_bytes"):
                store.load(reference)


if __name__ == "__main__":
    unittest.main()
