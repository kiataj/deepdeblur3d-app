"""Guards on the GUI stylesheet, checked as source text so CI needs no Qt."""
import pathlib
import re
import unittest

GUI = pathlib.Path(__file__).resolve().parents[1] / "src" / "deblur3d_app" / "gui.py"


def stylesheet(name: str) -> str:
    source = GUI.read_text(encoding="utf-8")
    match = re.search(rf'^{name} = """(.*?)"""', source, re.S | re.M)
    assert match, f"{name} not found in gui.py"
    return match.group(1)


class ReadoutStyleTests(unittest.TestCase):
    """napari themes the widgets through its own stylesheet, but `palette(...)`
    resolves against the Qt palette instead. Setting only one of background and
    text colour here can therefore land the text on a background of the same
    colour, which rendered the control fields blank for one reporter while
    looking fine on the developer's machine. Style the frame, nothing else.
    """

    def setUp(self):
        self.css = stylesheet("_READOUT_STYLE")

    def test_does_not_set_a_background(self):
        self.assertNotRegex(self.css, r"\bbackground\b")

    def test_does_not_set_a_text_colour(self):
        # `border: ... color` is fine; a bare `color:` property is not.
        declarations = [d.strip() for d in re.split(r"[;{}]", self.css)]
        self.assertFalse(
            [d for d in declarations if re.match(r"^color\s*:", d)],
            "readout must inherit its text colour from the active theme",
        )

    def test_does_not_resolve_colours_from_the_qt_palette(self):
        self.assertNotIn(
            "palette(", self.css,
            "palette(...) ignores the napari theme; use a theme-agnostic colour",
        )

    def test_still_draws_a_visible_frame(self):
        # The whole point of the style is making the field look editable.
        self.assertRegex(self.css, r"border\s*:")
        self.assertRegex(self.css, r"border-radius\s*:")


if __name__ == "__main__":
    unittest.main()
