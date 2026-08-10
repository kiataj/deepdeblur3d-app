import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from deblur3d_app import core


def fake_run(stdout="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


class BlockedReasonTests(unittest.TestCase):
    """The only thing an automatic update can destroy is uncommitted work."""

    def test_clean_checkout_is_allowed(self):
        with mock.patch.object(core, "_repo_root", return_value=Path("/repo")), \
             mock.patch.object(subprocess, "run", return_value=fake_run("")):
            self.assertIsNone(core.update_blocked_reason())

    def test_dirty_checkout_is_refused_and_lists_the_files(self):
        status = " M src/deblur3d_app/gui.py\n?? scratch.py\n"
        with mock.patch.object(core, "_repo_root", return_value=Path("/repo")), \
             mock.patch.object(subprocess, "run", return_value=fake_run(status)):
            reason = core.update_blocked_reason()
        self.assertIsNotNone(reason)
        self.assertIn("2 uncommitted change", reason)
        self.assertIn("gui.py", reason)

    def test_non_checkout_install_is_allowed(self):
        with mock.patch.object(core, "_repo_root", return_value=None):
            self.assertIsNone(core.update_blocked_reason())

    def test_missing_git_is_refused_rather_than_crashing(self):
        with mock.patch.object(core, "_repo_root", return_value=Path("/repo")), \
             mock.patch.object(subprocess, "run", side_effect=OSError("no git")):
            self.assertIn("git is not available", core.update_blocked_reason())

    def test_git_failure_is_refused(self):
        with mock.patch.object(core, "_repo_root", return_value=Path("/repo")), \
             mock.patch.object(subprocess, "run", return_value=fake_run("", 128)):
            self.assertIn("could not read", core.update_blocked_reason())


class UpdateCommandTests(unittest.TestCase):
    def test_checkout_pulls_fast_forward_only(self):
        with mock.patch.object(core, "_repo_root", return_value=Path("/repo")):
            cmds = core.update_commands("v9.9.9")
        # --ff-only so a diverged checkout fails loudly instead of merging.
        self.assertIn("--ff-only", cmds[0])
        self.assertEqual(cmds[0][:2], ["git", "-C"])
        self.assertEqual(cmds[1][:4], [sys.executable, "-m", "pip", "install"])

    def test_plain_install_pulls_the_tag_not_pypi(self):
        # deblur3d-gui is not published to PyPI, so a bare `pip install
        # --upgrade deblur3d-gui` fails with no matching distribution.
        with mock.patch.object(core, "_repo_root", return_value=None):
            cmds = core.update_commands("v9.9.9")
        self.assertEqual(len(cmds), 1)
        spec = cmds[0][-1]
        self.assertIn("git+https://github.com/", spec)
        self.assertTrue(spec.endswith("@v9.9.9"))

    def test_plain_install_falls_back_to_main_without_a_tag(self):
        with mock.patch.object(core, "_repo_root", return_value=None):
            self.assertTrue(core.update_commands()[0][-1].endswith("@main"))

    def test_instructions_match_the_commands(self):
        with mock.patch.object(core, "_repo_root", return_value=None):
            self.assertIn("git+https://github.com/", core.update_instructions("v1.2.3"))


class PerformUpdateTests(unittest.TestCase):
    class FakeProc:
        def __init__(self, lines, code=0):
            self.stdout = iter(lines)
            self._code = code
            self.returncode = code

        def wait(self):
            return self._code

    def test_streams_output_and_reports_success(self):
        seen = []
        with mock.patch.object(core, "update_commands", return_value=[["a"], ["b"]]), \
             mock.patch.object(subprocess, "Popen",
                               side_effect=lambda *a, **k: self.FakeProc(["one\n", "two\n"])):
            ok = core.perform_update(on_output=seen.append)
        self.assertTrue(ok)
        self.assertIn("one", seen)
        self.assertEqual(sum(1 for s in seen if s.startswith("$ ")), 2)

    def test_stops_at_the_first_failing_command(self):
        calls = []

        def popen(cmd, *a, **k):
            calls.append(cmd)
            return self.FakeProc([], code=1)

        with mock.patch.object(core, "update_commands", return_value=[["a"], ["b"]]), \
             mock.patch.object(subprocess, "Popen", side_effect=popen):
            ok = core.perform_update(on_output=lambda _l: None)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 1, "must not run later steps after a failure")

    def test_unlaunchable_command_fails_cleanly(self):
        seen = []
        with mock.patch.object(core, "update_commands", return_value=[["a"]]), \
             mock.patch.object(subprocess, "Popen", side_effect=OSError("boom")):
            self.assertFalse(core.perform_update(on_output=seen.append))
        self.assertTrue(any("could not run" in s for s in seen))


if __name__ == "__main__":
    unittest.main()
