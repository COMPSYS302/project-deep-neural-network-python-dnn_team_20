from PyQt5.QtWidgets import QMainWindow, QTabWidget
from gui.load_data_tab import LoadDataTab
from gui.view_data_tab import ViewDataTab

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 1000, 500)

                # Create the QTabWidget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create the tabs
        self.load_data_tab = LoadDataTab()
        self.view_data_tab = ViewDataTab()

        # Add tabs to the tab widget
        self.tabs.addTab(self.load_data_tab, "Load Data")
        self.tabs.addTab(self.view_data_tab, "View Data")

        # Connect signals (assuming you have this signal in LoadDataTab)
        self.load_data_tab.data_loaded.connect(self.view_data_tab.load_images)

       

