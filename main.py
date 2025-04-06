import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import SignLanguageApp

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load and apply the stylesheet
    with open('style.qss', 'r') as file:
        app.setStyleSheet(file.read())

    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
