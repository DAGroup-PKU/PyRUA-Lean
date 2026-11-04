"""Expert harness: helpers, adapter validation, brief rendering/lint, static audit, manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datafarm.common import append_jsonl
from datafarm.expert.adapter import (HoldoutSeed, SeedSet, TaskAdapter, load_holdout_seeds,
                                          validate_holdout_against_dev, write_holdout_seeds)
from datafarm.expert.audit import audit_expert_source
from datafarm.expert.brief import lint_brief, render_brief
from datafarm.expert.manifest import build_manifest, diff_manifests
from datafarm.expert.util import (ConfigError, UnsafePathError, expand,
                                       human_duration, read_jsonl, read_regular_nofollow, safe_relpath,
                                       sha256_json, tree_digest)
from xh_helpers import HOLDOUT, TEMPLATE, expert, write_adapter

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_ADAPTER = REPO / "configs" / "expert-harness" / "template.json"


class UtilTests(unittest.TestCase):
    def test_canonical_digest_is_order_independent(self) -> None:
        self.assertEqual(sha256_json({"a": 1, "b": [1, 2]}), sha256_json({"b": [1, 2], "a": 1}))

    def test_expand_placeholders_fail_closed(self) -> None:
        self.assertEqual(expand("{a}/{b}", {"a": "x", "b": 2}), "x/2")
        self.assertEqual(expand("{{literal}}", {}), "{literal}")
        self.assertEqual(expand("{env:HOME}/.cache", {}, keep_env=True), "${HOME}/.cache")
        with self.assertRaises(ConfigError):
            expand("{missing}", {})

    def test_human_duration(self) -> None:
        self.assertEqual(human_duration(17100), "4 h 45 min")
        self.assertEqual(human_duration(3600), "1 h")
        self.assertEqual(human_duration(600), "10 min")

    def test_jsonl_append_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l.jsonl"
            append_jsonl(path, {"a": 1})
            append_jsonl(path, {"a": 2})
            self.assertEqual([r["a"] for r in read_jsonl(path)], [1, 2])

    def test_safe_relpath_rejects_escapes(self) -> None:
        self.assertEqual(safe_relpath("./a//b.py"), "a/b.py")
        for bad in ("/etc/passwd", "../x", "a/../../x", "", "~/x", 3):
            with self.assertRaises(UnsafePathError):
                safe_relpath(bad)

    def test_nofollow_read_rejects_symlinks_anywhere(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            (root / "sub").mkdir(parents=True)
            secret = Path(tmp) / "secret.txt"
            secret.write_text("secret")
            (root / "ok.py").write_text("x = 1\n")
            (root / "link.py").symlink_to(secret)
            (root / "dirlink").symlink_to(Path(tmp))
            self.assertEqual(read_regular_nofollow(root, "ok.py", max_bytes=100), b"x = 1\n")
            with self.assertRaises(UnsafePathError):
                read_regular_nofollow(root, "link.py", max_bytes=100)
            with self.assertRaises(UnsafePathError):
                read_regular_nofollow(root, "dirlink/secret.txt", max_bytes=100)
            with self.assertRaises(UnsafePathError):
                read_regular_nofollow(root, "sub", max_bytes=100)
            with self.assertRaises(UnsafePathError):
                read_regular_nofollow(root, "ok.py", max_bytes=2)

    def test_tree_digest_changes_with_content_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("1")
            first = tree_digest(root)["sha256"]
            (root / "a.py").write_text("2")
            second = tree_digest(root)["sha256"]
            (root / "l").symlink_to("/etc")
            third = tree_digest(root)["sha256"]
            self.assertEqual(len({first, second, third}), 3)


class AdapterTests(unittest.TestCase):
    def test_template_adapter_is_valid(self) -> None:
        adapter = TaskAdapter.load(TEMPLATE_ADAPTER)
        self.assertEqual(adapter.task_id, "example-task")
        self.assertEqual(adapter.verifier_version, "v1")
        self.assertEqual(adapter.budgets.max_concurrent_rollouts, 6)
        self.assertIn(999, adapter.dev_seeds)
        self.assertNotIn(1000, adapter.dev_seeds)
        self.assertEqual(adapter.isolation("agent")["base"], "allowlist")
        self.assertIn("{run_dir}/expert_snapshot.py:{expert_class}", adapter.rollout["sim"]["argv"])
        self.assertTrue(any(p.endswith("run_rollout.py") for p in adapter.manifest_paths()))

    def test_invalid_configs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_adapter(root)
            raw = json.loads((root / "adapter.json").read_text())
            cases = {
                "schema": dict(raw, schema="nope"),
                "backend": dict(raw, agent=dict(raw["agent"], backend="gpt")),
                "budget": dict(raw, budgets=dict(raw["budgets"], max_counted_rollouts=0)),
                "concurrency": dict(raw, budgets=dict(raw["budgets"], max_concurrent_rollouts=99)),
                "wait_cap": dict(raw, budgets=dict(raw["budgets"], wait_cap_s=600)),
                "judge_points": dict(raw, verifier=dict(raw["verifier"], judge_points=[])),
                "criteria": dict(raw, verifier=dict(raw["verifier"], criteria=[])),
                "isolation": dict(raw, isolation={"agent": {"mode": "docker"}, "rollout": {"mode": "none"}}),
                "relative_path": dict(raw, workspace={"readable": ["relative/path"], "protected": []}),
                "seeds": dict(raw, seeds={"dev": {"ranges": [[5, 1]]}}),
            }
            for name, bad in cases.items():
                with self.subTest(name), self.assertRaises(ConfigError):
                    TaskAdapter(bad, root / "adapter.json")

    def test_typed_task_ir_is_validated_against_the_judge(self) -> None:
        from test_expert_harness_task_ir import hanging_ir
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_adapter(root)
            raw = json.loads((root / "adapter.json").read_text())
            typed = dict(hanging_ir(), profile="stage-graph/1")
            verifier = dict(raw["verifier"], criterion_ids=["c1"])
            TaskAdapter(dict(raw, task_ir=typed, verifier=verifier), root / "adapter.json")
            broken = dict(typed, stage_graph={"stages": [], "sync": []})
            with self.assertRaisesRegex(ConfigError, "task_ir stage graph is invalid"):
                TaskAdapter(dict(raw, task_ir=broken, verifier=verifier), root / "adapter.json")
            with self.assertRaisesRegex(ConfigError, "task_ir stage graph is invalid"):
                TaskAdapter(dict(raw, task_ir=typed, verifier=dict(verifier, criterion_ids=["c1", "c9"])),
                            root / "adapter.json")

    def test_seed_sets_and_holdout_files(self) -> None:
        seeds = SeedSet.from_config({"ranges": [[0, 9]], "explicit": [42]})
        self.assertIn(9, seeds)
        self.assertIn(42, seeds)
        self.assertNotIn(10, seeds)
        self.assertNotIn(True, seeds)
        self.assertEqual(seeds.describe(), "0–9, 42")
        with tempfile.TemporaryDirectory() as tmp:
            text = Path(tmp) / "seeds.txt"
            text.write_text("3002 family_a\n3000 family_b\n")
            loaded = load_holdout_seeds(text)
            self.assertEqual([(s.seed, s.family) for s in loaded], [(3002, "family_a"), (3000, "family_b")])
            path = Path(tmp) / "h.json"
            write_holdout_seeds(path, [HoldoutSeed(5, "x"), HoldoutSeed(1005, "x")], note="t")
            self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
            with self.assertRaises(ConfigError):
                validate_holdout_against_dev(load_holdout_seeds(path), seeds)


class BriefTests(unittest.TestCase):
    def test_template_brief_renders_and_lints(self) -> None:
        adapter = TaskAdapter.load(TEMPLATE_ADAPTER)
        text = render_brief(adapter)
        report = lint_brief(text, adapter, holdout_seeds=[123456, 234567])
        self.assertTrue(report.ok, report.errors)
        for needle in ("<minutes>", "xh flag-for-human", "$60", "80 counted development rollouts"):
            self.assertIn(needle, text)

    def test_each_missing_section_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = write_adapter(Path(tmp))
            text = render_brief(adapter)
            self.assertTrue(lint_brief(text, adapter).ok, lint_brief(text, adapter).errors)
            for heading in ("## Task", "## Expert interface", "## Verifier", "## Runtime limits", "## Budget",
                            "## Tools", "## Readable paths", "## Forbidden", "## Deliverables",
                            "## Submit protocol", "## Holdout policy", "## How to stop or abort"):
                broken = "\n".join(line for line in text.splitlines()
                                   if not line.startswith(heading))
                with self.subTest(heading):
                    self.assertFalse(lint_brief(broken, adapter).ok)

    def test_lint_catches_brief_defects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = write_adapter(Path(tmp))
            text = render_brief(adapter)
            cases = {
                "runtime fact": text.replace("100 samples", "a few samples"),
                "judge point": text.replace("settled", "final"),
                "budget number": text.replace("6 counted", "60 counted"),
                "holdout leak": text + "\nTry seed 5003 later.\n",
                "placeholder": text + "\n{{ unresolved }}\n",
                "abort channel": text.replace("xh flag-for-human", "ask someone"),
                "readable protected": text.replace("## Readable paths",
                                                   f"## Readable paths\n\n- `{tmp}/protected`"),
            }
            for name, broken in cases.items():
                with self.subTest(name):
                    self.assertFalse(lint_brief(broken, adapter, holdout_seeds=list(HOLDOUT)).ok)

    def test_template_placeholders_must_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = write_adapter(Path(tmp), template=TEMPLATE + "\n{{ facts.missing }}\n")
            from datafarm.expert.brief import BriefError

            with self.assertRaises(BriefError):
                render_brief(adapter)

    def test_masked_path_under_readable_must_be_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = write_adapter(root, {"workspace": {"protected": ["{root}/protected",
                                                                       "{root}/readable/secret.txt"]}})
            text = render_brief(adapter)
            self.assertTrue(lint_brief(text, adapter).ok)
            stripped = text.replace(f"{root}/readable/secret.txt", "(omitted)")
            self.assertFalse(lint_brief(stripped, adapter).ok)


class AuditTests(unittest.TestCase):
    POLICY = {"allowed_imports": ["__future__", "math", "numpy", "scipy", "example_backend.piper_ik"],
              "max_bytes": 10_000}

    def audit(self, source: str, cls: str | None = "E"):
        return audit_expert_source(source, cls, self.POLICY)

    def test_plain_expert_passes(self) -> None:
        report = self.audit("import math\nimport numpy as np\nfrom scipy.optimize import least_squares\n"
                            "from example_backend.piper_ik import NativePiperIK\n"
                            "class E:\n    def f(self, o):\n        return getattr(o, 'cloth', None), np.zeros(3)\n")
        self.assertTrue(report.ok, report.violations)

    def test_violations(self) -> None:
        cases = {
            "os import": "import os\nclass E: pass\n",
            "omni import": "import omni.isaac\nclass E: pass\n",
            "relative": "from . import x\nclass E: pass\n",
            "open": "class E:\n    def f(self):\n        return open('/etc/passwd').read()\n",
            "eval": "class E:\n    x = eval('1')\n",
            "np.load": "import numpy as np\nclass E:\n    x = np.load('a.npy')\n",
            "np.lib chain": "import numpy as np\nclass E:\n    x = np.lib.npyio.loadtxt('a')\n",
            "from numpy import load": "from numpy import load\nclass E: pass\n",
            "dunder getattr": "class E:\n    x = getattr(object, '__subclasses__')\n",
            "frame": "import math\nclass E:\n    def f(self, t):\n        return t.tb_frame\n",
            "missing class": "class F: pass\n",
            "syntax": "class E(:\n",
            "size": "class E: pass\n" + "#" * 20_000,
        }
        for name, source in cases.items():
            with self.subTest(name):
                self.assertFalse(self.audit(source).ok)

    def test_fixture_expert_passes(self) -> None:
        self.assertTrue(audit_expert_source(expert(0.5), "ToyExpert", {"allowed_imports": ["math"]}).ok)


class ManifestTests(unittest.TestCase):
    def test_detects_modification_deletion_and_addition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "judge.py").write_text("v1")
            (root / "pkg").mkdir()
            (root / "pkg" / "a.py").write_text("a")
            entries = [str(root / "judge.py"), str(root / "pkg"), str(root / "absent.json")]
            base = build_manifest(entries, extra={"code": "x"})
            self.assertEqual(diff_manifests(base, build_manifest(entries, extra={"code": "x"})), [])
            (root / "judge.py").write_text("v2")
            (root / "pkg" / "b.py").write_text("b")
            (root / "absent.json").write_text("{}")
            changed = {c["path"] for c in diff_manifests(base, build_manifest(entries, extra={"code": "y"}))}
            self.assertEqual(changed, set(entries) | {"extra:code"})


if __name__ == "__main__":
    unittest.main()
