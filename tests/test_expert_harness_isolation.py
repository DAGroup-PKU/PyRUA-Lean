"""Expert harness isolation: command construction (hermetic) and real bwrap / unshare smoke
tests showing that masked paths stay unreadable even after an ``umount`` attempt (skipped
when the host cannot create the sandbox)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from datafarm.expert.isolation import (IsolationUnavailable, SandboxSpec, bwrap_argv, job_wrapper_bash,
                                            make_empty_file, parse_probe, plan_mounts, probe_available,
                                            resolve_spec, run_probe, sandbox_env, unshare_argv, wrap)
from datafarm.expert.util import HarnessError
from xh_helpers import is_root

EXISTING = {"/data", "/data/allowed", "/data/allowed/secret.txt", "/data/allowed/sub", "/data/hidden",
            "/ws", "/round/tmp", "/opt/auth.json", "/srv/a/b/c.json"}
DIRS = {"/data", "/data/allowed", "/data/allowed/sub", "/data/hidden", "/ws", "/round/tmp"}


def fake_spec(**changes) -> SandboxSpec:
    spec = SandboxSpec(mode="bwrap", base="allowlist", ro=("/data/allowed", "/srv/a/b/c.json", "/missing"),
                       rw=("/ws",), masked=("/data/allowed/secret.txt", "/data/allowed/sub", "/data/hidden"),
                       binds=(("/round/tmp", "/tmp", True), ("/opt/auth.json", "/ws/.auth.json", False)),
                       system_ro=("/usr", "/etc"), cwd="/ws", empty_file="/round/.xh-empty")
    return spec.replace(**changes)


def argv(spec: SandboxSpec) -> list[str]:
    return bwrap_argv(spec, ["agent", "--flag"], lexists=lambda p: p in EXISTING or p in ("/usr", "/etc"),
                      isdir=lambda p: p in DIRS)


class CommandTests(unittest.TestCase):
    def test_masks_follow_their_readable_parent(self) -> None:
        ops = plan_mounts(fake_spec(), lexists=lambda p: p in EXISTING, isdir=lambda p: p in DIRS)
        order = [(op.kind, op.dest) for op in ops]
        self.assertLess(order.index(("ro", "/data/allowed")), order.index(("mask_file", "/data/allowed/secret.txt")))
        self.assertLess(order.index(("ro", "/data/allowed")), order.index(("mask_dir", "/data/allowed/sub")))
        self.assertNotIn(("ro", "/missing"), order)
        # never bound in an allowlist sandbox, so no mount point is created for it
        self.assertNotIn(("mask_dir", "/data/hidden"), order)
        # synthetic parents get explicit 0755 directories (bwrap would create them 0700)
        self.assertIn(("dir", "/srv"), order)
        self.assertIn(("dir", "/srv/a/b"), order)
        self.assertLess(order.index(("dir", "/srv/a/b")), order.index(("ro", "/srv/a/b/c.json")))

    def test_parent_missing_from_a_remapped_bind_is_created_readable(self) -> None:
        spec = fake_spec(ro=("/tmp/checkout/fakes",))
        ops = plan_mounts(spec, lexists=lambda p: p in EXISTING or p == "/tmp/checkout/fakes",
                          isdir=lambda p: p in DIRS)
        order = [(op.kind, op.dest) for op in ops]
        self.assertLess(order.index(("rw", "/tmp")), order.index(("dir", "/tmp/checkout")))
        self.assertLess(order.index(("dir", "/tmp/checkout")), order.index(("ro", "/tmp/checkout/fakes")))
        present = {"/tmp/checkout/fakes", "/round/tmp/checkout"}
        ops = plan_mounts(spec, lexists=lambda p: p in EXISTING or p in present, isdir=lambda p: p in DIRS)
        self.assertNotIn(("dir", "/tmp/checkout"), [(op.kind, op.dest) for op in ops])

    def test_bwrap_argv_drops_capabilities_and_keeps_secrets_out(self) -> None:
        cmd = argv(fake_spec())
        self.assertEqual(cmd[0], "bwrap")
        self.assertIn("--die-with-parent", cmd)
        self.assertIn("--unshare-pid", cmd)
        self.assertEqual(cmd[cmd.index("--cap-drop") + 1], "ALL")
        self.assertIn("--ro-bind-try", cmd)
        joined = " ".join(cmd)
        self.assertNotIn("--ro-bind / /", joined)
        self.assertIn("--bind /round/tmp /tmp", joined)
        self.assertIn("--ro-bind /opt/auth.json /ws/.auth.json", joined)
        self.assertIn("--ro-bind /round/.xh-empty /data/allowed/secret.txt", joined)
        self.assertIn("--tmpfs /data/allowed/sub", joined)
        self.assertIn("--remount-ro /data/allowed/sub", joined)
        self.assertEqual(cmd[-3:], ["--", "agent", "--flag"])
        self.assertNotIn("ANTHROPIC", joined)

    def test_run_as_uses_setpriv_and_minimal_caps(self) -> None:
        cmd = argv(fake_spec(run_as=(65534, 65534)))
        self.assertNotIn("--cap-drop", cmd)
        tail = cmd[cmd.index("--") + 1:]
        self.assertTrue(tail[0].endswith("setpriv"))
        self.assertIn("--reuid=65534", tail)
        self.assertIn("--bounding-set=-all", tail)
        self.assertIn("--no-new-privs", tail)
        self.assertEqual(tail[-2:], ["agent", "--flag"])

    def test_dangerous_capabilities_are_refused(self) -> None:
        with self.assertRaises(HarnessError):
            argv(fake_spec(keep_caps=("CAP_DAC_READ_SEARCH",)))
        with self.assertRaises(HarnessError):
            argv(fake_spec(keep_caps=("CAP_SYS_ADMIN",)))
        self.assertIn("CAP_DAC_OVERRIDE", argv(fake_spec(keep_caps=("CAP_DAC_OVERRIDE",))))

    def test_root_ro_base_and_host_devices(self) -> None:
        cmd = argv(fake_spec(base="root_ro", dev="host", share_net=False))
        joined = " ".join(cmd)
        self.assertIn("--ro-bind / /", joined)
        self.assertIn("--dev-bind /dev /dev", joined)
        self.assertIn("--unshare-net", cmd)
        self.assertIn("--tmpfs /data/hidden", joined)  # root_ro: every existing protected path is masked

    def test_unshare_fallback(self) -> None:
        with self.assertRaises(IsolationUnavailable):
            unshare_argv(fake_spec(mode="unshare"), ["x"], lexists=lambda p: p in EXISTING, isdir=lambda p: p in DIRS)
        cmd = unshare_argv(fake_spec(mode="unshare", base="root_ro"), ["agent"],
                           lexists=lambda p: p in EXISTING, isdir=lambda p: p in DIRS)
        self.assertEqual(cmd[:4], ["unshare", "--mount", "--propagation", "private"])
        script = cmd[cmd.index("-c") + 1]
        self.assertIn("mount -t tmpfs -o ro,size=16k,mode=0555 tmpfs /data/hidden", script)
        self.assertIn("mount -o remount,bind,rw /ws", script)
        self.assertIn("--bounding-set=-all", script)
        self.assertEqual(cmd[-1], "agent")

    def test_resolve_spec_falls_back_and_fails_closed(self) -> None:
        spec = fake_spec()
        only_unshare = lambda mode: (mode == "unshare", "n/a")  # noqa: E731
        with self.assertRaises(IsolationUnavailable):
            resolve_spec(spec, fallback="unshare", allow_degraded=False, available=only_unshare)
        degraded = resolve_spec(spec, fallback="unshare", allow_degraded=True, available=only_unshare)
        self.assertEqual((degraded.mode, degraded.base, degraded.degraded), ("unshare", "root_ro", True))
        with self.assertRaises(IsolationUnavailable):
            resolve_spec(spec, fallback=None, allow_degraded=True, available=lambda m: (False, "no"))

    def test_sandbox_env_passes_named_secrets_only(self) -> None:
        os.environ["XH_TEST_SECRET"] = "s3cret"
        os.environ["XH_TEST_OTHER"] = "other"
        try:
            env = sandbox_env({"HOME": "/h"}, ["XH_TEST_SECRET", "XH_TEST_ABSENT"])
        finally:
            del os.environ["XH_TEST_SECRET"], os.environ["XH_TEST_OTHER"]
        self.assertEqual(env, {"HOME": "/h", "XH_TEST_SECRET": "s3cret"})

    def test_probe_parsing(self) -> None:
        result = parse_probe(["HIDDEN /a", "LEAK /b", "UMOUNTED /c", "WRITABLE /ro", "READONLY /ws"],
                             mode="bwrap", readonly_dirs=["/ro"], writable_dirs=["/ws"])
        self.assertFalse(result.ok)
        self.assertEqual((result.leaks, result.umounted, result.writable_readonly, result.unwritable_workspace),
                         (["/b"], ["/c"], ["/ro"], ["/ws"]))
        self.assertTrue(parse_probe(["HIDDEN /a", "READONLY /ro", "WRITABLE /ws"], mode="bwrap",
                                    readonly_dirs=["/ro"], writable_dirs=["/ws"]).ok)

    def test_job_wrapper_checks_paths_at_run_time(self) -> None:
        text = job_wrapper_bash(mode="bwrap", fallback="unshare", base="root_ro", dev="host",
                                ro=["/run/dir"], rw=["/run/dir/native", "${HOME}/.cache"],
                                masked=["/secret", "${HOME}/.ssh"], run_as=None)
        self.assertIn('mask "${HOME}/.ssh"', text)
        self.assertIn("--cap-drop ALL", text)
        self.assertIn("--tmpfs /tmp", text)
        self.assertIn("mount -o remount,bind,rw", text)
        result = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        # operations are ordered at run time, after ${HOME} has expanded on the node
        script = text + '\nxh_op() { echo "$1 $2"; }\nxh_collect\n'
        out = subprocess.run(["bash", "-c", script], env={"HOME": "/h", "PATH": "/usr/bin:/bin"},
                             capture_output=True, text=True).stdout.splitlines()
        self.assertEqual(out, ["mask /secret", "ro /run/dir", "rw /h/.cache", "mask /h/.ssh",
                               "rw /run/dir/native"])


def _can(mode: str) -> bool:
    return probe_available(mode)[0]


class RealSandboxTests(unittest.TestCase):
    """Run harmless commands in real sandboxes; masked content must stay unreadable."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="xh-iso-", dir="/var/tmp" if os.access("/var/tmp", os.W_OK) else None)
        os.chmod(self.tmp, 0o755)
        root = Path(self.tmp)
        (root / "readable" / "sub").mkdir(parents=True)
        (root / "readable" / "ok.txt").write_text("fine\n")
        (root / "readable" / "secret.txt").write_text("TOP-SECRET\n")
        (root / "readable" / "sub" / "deep.txt").write_text("DEEP-SECRET\n")
        (root / "hidden").mkdir()
        (root / "hidden" / "x.txt").write_text("HIDDEN-SECRET\n")
        (root / "ws").mkdir()
        for path in (root, root / "readable", root / "readable" / "sub", root / "ws"):
            os.chmod(path, 0o755)
        self.root = root
        self.empty = make_empty_file(root / "private")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def spec(self, mode: str, base: str, **changes) -> SandboxSpec:
        r = self.root
        return SandboxSpec(mode=mode, base=base, ro=(str(r / "readable"),), rw=(str(r / "ws"),),
                           masked=(str(r / "readable" / "secret.txt"), str(r / "readable" / "sub"),
                                   str(r / "hidden")),
                           cwd=str(r / "ws"), empty_file=self.empty, share_net=False).replace(**changes)

    def run_in(self, spec: SandboxSpec, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(wrap(spec, ["/bin/sh", "-c", script]), capture_output=True, text=True, timeout=60,
                              env={"PATH": "/usr/bin:/bin"})

    def check_masks(self, spec: SandboxSpec) -> None:
        r = self.root
        script = (f"cat {r}/readable/ok.txt; "
                  f"for p in {r}/readable/sub {r}/readable/secret.txt {r}/hidden; do umount $p 2>/dev/null; done; "
                  f"cat {r}/readable/secret.txt {r}/readable/sub/deep.txt {r}/hidden/x.txt 2>/dev/null; "
                  f"echo w > {r}/ws/out.txt && echo WS-OK; "
                  f"echo w > {r}/readable/new.txt 2>/dev/null && echo RO-BROKEN; true")
        result = self.run_in(spec, script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fine", result.stdout)
        self.assertIn("WS-OK", result.stdout)
        for secret in ("TOP-SECRET", "DEEP-SECRET", "HIDDEN-SECRET", "RO-BROKEN"):
            self.assertNotIn(secret, result.stdout)
        probe = run_probe(spec, masked=[str(r / "readable" / "secret.txt"), str(r / "readable" / "sub"),
                                        str(r / "hidden")],
                          readonly_dirs=[str(r / "readable")], writable_dirs=[str(r / "ws")])
        self.assertTrue(probe.ok, probe.to_dict())

    @unittest.skipUnless(_can("bwrap"), "bwrap cannot create a sandbox here")
    def test_bwrap_allowlist_masks(self) -> None:
        self.check_masks(self.spec("bwrap", "allowlist"))

    @unittest.skipUnless(_can("bwrap"), "bwrap cannot create a sandbox here")
    def test_bwrap_root_ro_masks(self) -> None:
        self.check_masks(self.spec("bwrap", "root_ro"))

    @unittest.skipUnless(_can("bwrap") and is_root(), "needs root to switch to an unprivileged uid")
    def test_bwrap_run_as_nobody(self) -> None:
        os.chown(self.root / "ws", 65534, 65534)
        spec = self.spec("bwrap", "allowlist", run_as=(65534, 65534))
        self.check_masks(spec)
        result = self.run_in(spec, "id -u; grep CapEff /proc/self/status")
        self.assertIn("65534", result.stdout)
        self.assertIn("0000000000000000", result.stdout)

    @unittest.skipUnless(_can("bwrap") and is_root(), "needs root to switch to an unprivileged uid")
    def test_run_as_nobody_reads_a_host_path_under_the_remapped_tmp(self) -> None:
        host = Path(tempfile.mkdtemp(prefix="xh-iso-host-", dir="/tmp"))
        self.addCleanup(shutil.rmtree, host, True)
        os.chmod(host, 0o755)
        (host / "sub").mkdir()
        os.chmod(host / "sub", 0o755)
        (host / "sub" / "tool.txt").write_text("TOOL-OK\n")
        round_tmp = self.root / "round-tmp"
        round_tmp.mkdir()
        os.chown(round_tmp, 65534, 65534)
        spec = self.spec("bwrap", "allowlist", run_as=(65534, 65534), ro=(str(host / "sub"),),
                         binds=((str(round_tmp), "/tmp", True),))
        result = self.run_in(spec, f"cat {host}/sub/tool.txt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TOOL-OK", result.stdout)

    @unittest.skipUnless(_can("bwrap") and is_root(), "needs root: shows why capabilities are dropped")
    def test_capabilities_would_defeat_masks(self) -> None:
        """Documented threat: bwrap started by root keeps all capabilities unless told
        otherwise, and CAP_SYS_ADMIN can unmount a mask. The harness always drops them."""

        r = self.root
        raw = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", str(r / "hidden"),
               "--unshare-pid", "--die-with-parent", "--", "/bin/sh", "-c",
               f"umount {r}/hidden && cat {r}/hidden/x.txt"]
        leaked = subprocess.run(raw, capture_output=True, text=True, timeout=60)
        if "HIDDEN-SECRET" not in leaked.stdout:
            self.skipTest("this bwrap already drops capabilities by default")
        guarded = self.run_in(self.spec("bwrap", "root_ro"), f"umount {r}/hidden; cat {r}/hidden/x.txt")
        self.assertNotIn("HIDDEN-SECRET", guarded.stdout)

    @unittest.skipUnless(is_root() and _can("unshare"), "unshare -m needs root")
    def test_unshare_masks(self) -> None:
        self.check_masks(self.spec("unshare", "root_ro"))


if __name__ == "__main__":
    unittest.main()
