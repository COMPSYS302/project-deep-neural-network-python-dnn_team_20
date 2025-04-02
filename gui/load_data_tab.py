import os
import time
import threading
import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QProgressBar, QLabel, QFileDialog
from PyQt5.QtCore import pyqtSignal

class LoadDataTab(QWidget):
    data_loaded = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_loading)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)
        self.loading = False
        self.data_path = None

        self.progress_updated.connect(self.progress_bar.setValue)
        self.status_updated.connect(self.status_label.setText)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.load_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_updated.emit("Loading data...")
            self.loading = True

            self.thread = threading.Thread(target=self.process_data, args=(file_path,))
            self.thread.start()

    def process_data(self, file_path):
        try:
            image_folder = self.convert_csv_to_images(file_path)
            self.data_path = image_folder

            with open(file_path, 'r') as file:
                lines = file.readlines()
            total = len(lines)

            for i, line in enumerate(lines):
                if not self.loading:
                    self.status_updated.emit("Loading stopped.")
                    break
                time.sleep(0.01)
                self.progress_updated.emit(int((i / total) * 100))

            if self.loading:
                self.status_updated.emit("Data Loaded Successfully!")
                self.data_loaded.emit(self.data_path)
        finally:
            self.load_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.loading = False

    def stop_loading(self):
        self.loading = False

    def convert_csv_to_images(self, csv_file):
        image_folder = os.path.join(os.path.dirname(csv_file), "images")
        os.makedirs(image_folder, exist_ok=True)

        for i in range(36):
            class_folder = os.path.join(image_folder, str(i))
            if not os.listdir(class_folder):  # Folder is empty
                os.rmdir(class_folder)  # Remove it
            os.makedirs(class_folder, exist_ok=True)

        with open(csv_file, 'r') as file:
            lines = file.readlines()
            _ = lines[0].strip().split(',')  # ignore header

            for idx, line in enumerate(lines[1:]):
                values = line.strip().split(',')
                label = int(values[0])
                pixels = np.array(values[1:], dtype=np.uint8).reshape(28, 28)
                image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
                image_path = os.path.join(image_folder, str(label), f"{label}_{idx}.png")
                cv2.imwrite(image_path, image)

        return image_folder