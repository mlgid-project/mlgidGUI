"""Entry point of the frozen app.

PyInstaller freezes a script, not a ``[project.gui-scripts]`` entry, so this
is the script: it does what the ``mlgidgui`` command does.
"""

import sys

from mlgidgui import main

if __name__ == "__main__":
    sys.exit(main())
