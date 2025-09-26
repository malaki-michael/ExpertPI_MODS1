import IPython
from traitlets.config import Config

# doesn't work for new ipython versions
# import os
# os.environ["QT_API"] = "PySide6"

c = Config()
c.InteractiveShellApp.gui = "qt6"
c.TerminalIPythonApp.matplotlib = "qt6"
c.InteractiveShellApp.exec_lines = [
    "from expert_pi.__main__ import *",
]

IPython.start_ipython(config=c)
