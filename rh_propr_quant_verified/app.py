from __future__ import annotations

import sys
import traceback

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.main_window import MainWindow


def _set_safe_app_font(app: QApplication) -> None:
    font = QFont()
    font.setFamilies(["Segoe UI", "Arial", "Tahoma", "Verdana"])
    font.setPointSize(10)
    app.setFont(font)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _set_safe_app_font(app)
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as exc:
        traceback.print_exc()
        QMessageBox.critical(None, "Startup error", f"Application failed to start:\n\n{exc}\n\nSee console output for details.")
        sys.exit(1)
