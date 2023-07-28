import sys

if sys.version_info < (3, 10):
    raise RuntimeError("DNA Painter requires Python 3.10 or later")

from .mondrian import Mondrian, mondrianize  # noqa: F401
