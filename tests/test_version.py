import subprocess
import types
import unittest
from unittest import mock

from packaging.version import Version

from deblur3d_app import _guess_next, _version_from_git


def _describe(output: str):
    """Stub `git describe --long --tags --dirty --always` with a fixed result."""
    return mock.patch.object(
        subprocess, "run",
        return_value=types.SimpleNamespace(returncode=0, stdout=output + "\n"),
    )


class GuessNextTests(unittest.TestCase):
    def test_bumps_last_numeric_component(self):
        self.assertEqual(_guess_next("2.0.0"), "2.0.1")
        self.assertEqual(_guess_next("1.9"), "1.10")
        self.assertEqual(_guess_next("3"), "4")


class VersionFromGitTests(unittest.TestCase):
    def test_tagged_commit_is_the_bare_tag(self):
        with _describe("v2.0.0-0-gabc1234"):
            self.assertEqual(_version_from_git(), "2.0.0")

    def test_commits_past_tag_get_next_dev_version(self):
        with _describe("v2.0.0-7-gdeadbee"):
            self.assertEqual(_version_from_git(), "2.0.1.dev7+gdeadbee")

    def test_dirty_tree_is_marked(self):
        with _describe("v2.0.0-0-gabc1234-dirty"):
            self.assertEqual(_version_from_git(), "2.0.0+dirty")
        with _describe("v2.0.0-7-gdeadbee-dirty"):
            self.assertEqual(_version_from_git(), "2.0.1.dev7+gdeadbee.dirty")

    def test_untagged_history_falls_back_to_hash(self):
        with _describe("abc1234"):
            self.assertEqual(_version_from_git(), "0.0.0+gabc1234")
        with _describe("abc1234-dirty"):
            self.assertEqual(_version_from_git(), "0.0.0+gabc1234.dirty")

    def test_all_shapes_are_pep440_and_ordered(self):
        shapes = [
            "v2.0.0-0-gabc1234",
            "v2.0.0-0-gabc1234-dirty",
            "v2.0.0-7-gdeadbee",
            "v2.0.0-7-gdeadbee-dirty",
            "v1.9-3-gfeed123",
            "abc1234",
        ]
        for shape in shapes:
            with self.subTest(shape=shape), _describe(shape):
                Version(_version_from_git())  # raises InvalidVersion if malformed

        # A later commit must sort above the tag it descends from.
        with _describe("v2.0.0-0-gabc1234"):
            tagged = Version(_version_from_git())
        with _describe("v2.0.0-7-gdeadbee"):
            later = Version(_version_from_git())
        self.assertGreater(later, tagged)

    def test_failed_git_invocation_returns_none(self):
        with mock.patch.object(
            subprocess, "run",
            return_value=types.SimpleNamespace(returncode=128, stdout=""),
        ):
            self.assertIsNone(_version_from_git())

        with mock.patch.object(subprocess, "run", side_effect=OSError):
            self.assertIsNone(_version_from_git())


if __name__ == "__main__":
    unittest.main()
