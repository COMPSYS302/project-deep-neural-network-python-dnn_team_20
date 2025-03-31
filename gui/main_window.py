from PyQt5.QtWidgets import QMainWindow, QTabWidget
from gui.load_data_tab import LoadDataTab
from gui.view_data_tab import ViewDataTab

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 800, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.load_data_tab = LoadDataTab()
        self.view_data_tab = ViewDataTab()

        self.tabs.addTab(self.load_data_tab, "Load Data")
        self.tabs.addTab(self.view_data_tab, "View Data")

        self.load_data_tab.data_loaded.connect(self.view_data_tab.load_images)
