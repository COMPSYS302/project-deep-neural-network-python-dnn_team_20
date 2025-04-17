from PyQt5.QtWidgets import QMainWindow, QTabWidget, QSplitter, QWidget, QVBoxLayout, QSpacerItem, QSizePolicy, QLabel, QPushButton
from PyQt5.QtCore import Qt
from gui.home_tab import HomeTab
from gui.load_data_tab import LoadDataTab
from gui.view_data_tab import ViewDataTab
from gui.train_tab import TrainTab
from PyQt5.QtGui import QPixmap
from gui.test_tab import TestTab


class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignCare")

        self.resize(900, 650)  # Set the initial size of the window

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Horizontal)

        # Create the QTabWidget
        self.tabs = QTabWidget()
        # self.setCentralWidget(self.tabs)
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


        splitter.addWidget(self.tabs)
        # left panel takes 2/3 of the window

        # Right Menu panel
        self.menu_bar = QWidget()
        menu_layout = QVBoxLayout()
        self.menu_bar.setLayout(menu_layout)

        # Add buttons to the menu bar
        spacer_top = QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        menu_layout.addItem(spacer_top)

        image_label = QLabel()
        pixmap = QPixmap("gui/images/asl.png").scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(pixmap)  # Center the image in the label
        menu_layout.addWidget(image_label, alignment=Qt.AlignCenter) 

        self.load_button = QPushButton("LOAD")
        self.view_button = QPushButton("VIEW")
        self.train_button = QPushButton("TRAIN")
        self.test_button = QPushButton("TEST")

        self.load_button.clicked.connect(self.go_to_load_tab)
        self.view_button.clicked.connect(self.go_to_view_tab)
        self.train_button.clicked.connect(self.go_to_train_tab)
        self.test_button.clicked.connect(self.go_to_test_tab)

        menu_layout.addWidget(self.load_button)
        menu_layout.addItem(QSpacerItem(20, 5, QSizePolicy.Minimum, QSizePolicy.Fixed))  # Spacer with 40px vertical height
        menu_layout.addWidget(self.view_button)
        menu_layout.addItem(QSpacerItem(20, 5, QSizePolicy.Minimum, QSizePolicy.Fixed))  # Spacer with 40px vertical height
        menu_layout.addWidget(self.train_button)
        menu_layout.addItem(QSpacerItem(20, 5, QSizePolicy.Minimum, QSizePolicy.Fixed))  # Spacer with 40px vertical height
        menu_layout.addWidget(self.test_button) 
        

        spacer_bottom = QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        menu_layout.addItem(spacer_bottom)

        splitter.addWidget(self.menu_bar)
        splitter.setSizes([650, 250])

        # Set the splitter as the central widget
        self.setCentralWidget(splitter)

        # Connect signals (assuming you have this signal in LoadDataTab)
        self.load_data_tab.data_loaded.connect(self.view_data_tab.load_images)
        self.load_data_tab.data_loaded.connect(self.train_tab.set_dataset_path)

        self.tabs.setCurrentIndex(0)

    def go_to_load_tab(self):
        """Switch to the Load tab."""
        self.tabs.setCurrentIndex(1)

    def go_to_view_tab(self):
        """Switch to the View tab."""
        self.tabs.setCurrentIndex(2)

    def go_to_train_tab(self):
        """Switch to the Train tab."""
        self.tabs.setCurrentIndex(3)

    def go_to_test_tab(self):
        """Switch to the Test tab."""
        self.tabs.setCurrentIndex(4)
