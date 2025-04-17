from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy, QSpacerItem
from PyQt5.QtCore import Qt

class HomeTab(QWidget):
    def __init__(self, tab_widget):
        super().__init__()

        self.tab_widget = tab_widget

        layout = QVBoxLayout()

        spacer_top = QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer_top)
        
        self.title_label = QLabel("SignCare")
        self.title_label.setStyleSheet("font-size: 60px; font-weight: bold; color: #000000; text-align: center; margin-bottom: 20px;")  # 20px gap below title
        
        layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        self.load_button = QPushButton("LOAD")
        self.view_button = QPushButton("VIEW")
        self.train_button = QPushButton("TRAIN")
        self.test_button = QPushButton("TEST")

        self.load_button.setObjectName("load_button")
        self.view_button.setObjectName("view_button")
        self.train_button.setObjectName("train_button")
        self.test_button.setObjectName("test_button")

        self.load_button.clicked.connect(self.go_to_load_tab)
        self.view_button.clicked.connect(self.go_to_view_tab)
        self.train_button.clicked.connect(self.go_to_train_tab)
        self.test_button.clicked.connect(self.go_to_test_tab)

        self.load_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.view_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.train_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self.load_button)
        layout.addWidget(self.view_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.test_button)

        spacer_bottom = QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer_bottom)

        layout.setSpacing(10)

        # Set layout for the home tab widget
        self.setLayout(layout)

    def go_to_load_tab(self):
        """Switch to the Load tab."""
        self.tab_widget.setCurrentIndex(1)

    def go_to_view_tab(self):
        """Switch to the View tab."""
        self.tab_widget.setCurrentIndex(2)

    def go_to_train_tab(self):
        """Switch to the Train tab."""
        self.tab_widget.setCurrentIndex(3)

    def go_to_test_tab(self):
        """Switch to the Test tab."""
        self.tab_widget.setCurrentIndex(4)
