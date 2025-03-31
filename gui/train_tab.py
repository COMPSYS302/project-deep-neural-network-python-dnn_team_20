from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider
)
from PyQt5.QtCore import Qt


class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # Start Training Button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button)

        # Model Dropdown
        layout.addWidget(QLabel("Select Model"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["AlexNet", "InceptionV3", "CustomCNN"])
        layout.addWidget(self.model_dropdown)

        # Train/Test Split
        layout.addWidget(QLabel("Train/Test Ratio"))
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(50, 95)  # 50% to 95%
        self.split_slider.setValue(80)      # default 80%
        layout.addWidget(self.split_slider)

        # Batch Size
        layout.addWidget(QLabel("Batch size"))
        self.batch_slider = QSlider(Qt.Horizontal)
        self.batch_slider.setRange(8, 128)
        self.batch_slider.setValue(32)
        layout.addWidget(self.batch_slider)

        # Epochs
        layout.addWidget(QLabel("Epochs"))
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setRange(1, 100)
        self.epoch_slider.setValue(30)
        layout.addWidget(self.epoch_slider)

        self.setLayout(layout)

    def start_training(self):
        # Collect values from UI
        model_name = self.model_dropdown.currentText()
        split_ratio = self.split_slider.value()
        batch_size = self.batch_slider.value()
        epochs = self.epoch_slider.value()

        print(f"Training {model_name} | Split: {split_ratio}% | Batch: {batch_size} | Epochs: {epochs}")

        # TODO: Pass to training function
