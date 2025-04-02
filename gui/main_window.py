from PyQt5.QtWidgets import QMainWindow, QTabWidget
from gui.load_data_tab import LoadDataTab
from gui.view_data_tab import ViewDataTab
from gui.train_tab import TrainTab


class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 800, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.load_data_tab = LoadDataTab()
        self.view_data_tab = ViewDataTab()
        self.train_tab = TrainTab()

        self.tabs.addTab(self.load_data_tab, "Load Data")
        self.tabs.addTab(self.view_data_tab, "View Data")
        self.tabs.addTab(self.train_tab, "Train")

        self.load_data_tab.data_loaded.connect(self.view_data_tab.load_images)
        self.load_data_tab.data_loaded.connect(self.train_tab.set_dataset_path)
