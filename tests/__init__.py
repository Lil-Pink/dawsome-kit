# The scripts live in tools/ and import each other by name; -I keeps the working folder off
# sys.path, so the tests add exactly that folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
