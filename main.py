"""Pantheon v0.4 — start with: python main.py"""
import sys
from PySide6.QtWidgets import QApplication
from brain.simulation import Brain
from interface.window import PantheonWindow


def main():
    app = QApplication(sys.argv)
    brain = Brain()
    window = PantheonWindow(brain)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
