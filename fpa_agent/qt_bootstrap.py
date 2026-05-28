"""
Configure Qt plugin paths before OpenCV is imported.

OpenCV (opencv-python) ships its own Qt plugins; if they load first, PyQt5
fails with: "Could not load the Qt platform plugin xcb" from cv2/qt/plugins.
"""

import os
import sys


def configure_qt_for_pyqt5():
    # Remove paths OpenCV may have set (or that point at cv2's bundled Qt).
    for key in (
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_PLUGIN_PATH",
    ):
        os.environ.pop(key, None)

    try:
        import PyQt5

        qt_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
        if os.path.isdir(qt_plugins):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugins
        # Prefer system/platform plugin from PyQt5, not OpenCV's copy.
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    except ImportError:
        pass

    # Tell OpenCV not to hijack Qt when built with Qt support.
    os.environ["OPENCV_VIDEOIO_PRIORITY_LIST"] = os.environ.get(
        "OPENCV_VIDEOIO_PRIORITY_LIST", "FFMPEG,GSTREAMER,V4L2"
    )


def ensure_qt_bootstrap():
    if not getattr(ensure_qt_bootstrap, "_done", False):
        configure_qt_for_pyqt5()
        ensure_qt_bootstrap._done = True


# Run on import of this module (call early from detection.py).
configure_qt_for_pyqt5()
