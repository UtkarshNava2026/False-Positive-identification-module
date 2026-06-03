"""Entry point — configure Qt before OpenCV, then start the PyQt5 app."""
import torch  # Import torch first to avoid WinError 1114 DLL loading conflicts with PyQt5/OpenCV
import importlib.util
import sys
from pathlib import Path


def _bootstrap_qt():
    """Load qt_bootstrap without executing fpa_agent/__init__.py."""
    path = Path(__file__).resolve().parent / "fpa_agent" / "qt_bootstrap.py"
    spec = importlib.util.spec_from_file_location("fpa_qt_bootstrap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


_bootstrap_qt()

from PyQt5.QtWidgets import QApplication

from fpa_agent.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
