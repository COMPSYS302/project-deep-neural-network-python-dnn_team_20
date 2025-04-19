from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy, QSpacerItem, QTextEdit
from PyQt5.QtCore import Qt

class HomeTab(QWidget):
    def __init__(self, tab_widget):
        super().__init__()

        self.tab_widget = tab_widget

        layout = QVBoxLayout()

        spacer_top = QSpacerItem(20, 50, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer_top)
        
        self.title_label = QLabel("SignCare")
        self.title_label.setObjectName("title_label")
        layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        # Blue Box with Catch Phrase
        self.catch_phrase_box = QLabel(" seamless communication between Deaf patients and healthcare providers")
        self.catch_phrase_box.setObjectName("catch_phrase_box")
        self.catch_phrase_box.setAlignment(Qt.AlignCenter)
        self.catch_phrase_box.setWordWrap(True)

        self.catch_phrase_box.setFixedHeight(65)  # Set height of the blue box
        self.catch_phrase_box.setFixedWidth(400)
        layout.addWidget(self.catch_phrase_box, alignment=Qt.AlignCenter)

        # Bottom spacer for spacing below the box
        spacer_bottom = QSpacerItem(20, 150, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer_bottom)

        # Set layout spacing
        layout.setSpacing(10)

        # Set layout for the home tab widget
        self.setLayout(layout)
