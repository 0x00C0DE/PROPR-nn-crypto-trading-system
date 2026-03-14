from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def _set_safe_app_font(app: QApplication) -> None:
    font = QFont()
    font.setFamilies(["Segoe UI", "Arial", "Tahoma", "Verdana"])
    font.setPointSize(10)
    app.setFont(font)


def main() -> int:
    app = QApplication(sys.argv)
    _set_safe_app_font(app)
    window = MainWindow()
    window.show()
    app.processEvents()

    out = Path(__file__).resolve().parent / 'gui_preview.png'

    def capture() -> None:
        app.processEvents()
        pixmap = QPixmap(window.size())
        window.render(pixmap)
        pixmap.save(str(out))
        print(f'Saved {out}')
        app.quit()

    QTimer.singleShot(600, capture)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
