from PyQt5.QtWidgets import QMainWindow, QTabWidget
from gui.home_tab import HomeTab
from gui.load_data_tab import LoadDataTab
from gui.view_data_tab import ViewDataTab
from gui.train_tab import TrainTab
from gui.test_tab import TestTab


class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignCare")

        # Create the QTabWidget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create the tabs
        self.home_tab = HomeTab(self.tabs)
        self.load_data_tab = LoadDataTab()
        self.view_data_tab = ViewDataTab()
        self.train_tab = TrainTab()
        self.test_tab = TestTab(self.train_tab)

        # Add tabs to the tab widget
        self.tabs.addTab(self.home_tab, "HOME")
        self.tabs.addTab(self.load_data_tab, "LOAD")
        self.tabs.addTab(self.view_data_tab, "VIEW")
        self.tabs.addTab(self.train_tab, "TRAIN")
        self.tabs.addTab(self.test_tab, "TEST")


        # Connect signals (assuming you have this signal in LoadDataTab)
        self.load_data_tab.data_loaded.connect(self.view_data_tab.load_images)
        self.load_data_tab.data_loaded.connect(self.train_tab.set_dataset_path)

        self.tabs.setCurrentIndex(0)
