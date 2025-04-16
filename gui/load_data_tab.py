import os
import time
import threading
import numpy as np
import cv2
import shutil  # Add this import
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QProgressBar, QLabel, QFileDialog
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel, QWidget, QSpacerItem, QSizePolicy

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import *

class LoadDataTab(QWidget):
    # Define custom signals for emitting events
    data_loaded = pyqtSignal(str)  # Emitted when data is successfully loaded
    progress_updated = pyqtSignal(int)  # Emitted to update progress bar
    status_updated = pyqtSignal(str)  # Emitted to update the status message

    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()

        # Create a QLabel for the title
        self.title_label = QLabel("SignCare")
        self.title_label.setStyleSheet("font-size: 60px; font-weight: bold; color: #000000; text-align: center; margin-bottom: 20px;")
        
        # Add the title label to the layout
        layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        # Button to load data (CSV file)
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)  # Connect to load_data method
        layout.addWidget(self.load_button)

        # Progress bar to show loading progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Label to display loading status
        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        # Button to stop the loading process
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)  # Initially disabled
        self.stop_button.clicked.connect(self.stop_loading)  # Connect to stop_loading method
        layout.addWidget(self.stop_button)

        self.setLayout(layout)
        self.loading = False  # To track if loading is in progress
        self.data_path = None  # To store the path of the loaded data

        # Connect signals to their respective slots
        self.progress_updated.connect(self.progress_bar.setValue)
        self.status_updated.connect(self.status_label.setText)

    def load_data(self):
        """Method to open a file dialog and select a CSV file to load data."""
        # Open file dialog to select CSV file
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)")

        
        if file_paths:
            self.load_button.setEnabled(False)  # Disable the load button
            self.stop_button.setEnabled(True)  # Enable the stop button
            self.status_updated.emit("Loading data...")  # Update status message
            self.loading = True  # Set loading to True

            # Start the data processing in a separate thread
            self.thread = threading.Thread(target=self.process_data, args=(file_paths,))
            self.thread.start()

    def process_data(self, file_paths):
        try:
            # Create a shared image folder for all files
            timestamp_folder = os.path.join(os.path.dirname(file_paths[0]), "images_" + str(int(time.time())))
            os.makedirs(timestamp_folder, exist_ok=True)
            for i in range(36):
                os.makedirs(os.path.join(timestamp_folder, str(i)), exist_ok=True)

            self.data_path = timestamp_folder

            total_lines = 0
            all_lines = []

            # Read and count all lines first (for progress bar)
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    lines = file.readlines()[1:]  # Skip header
                    all_lines.append((file_path, lines))
                    total_lines += len(lines)

            processed_lines = 0

            for file_path, lines in all_lines:
                self.convert_csv_to_images(file_path, timestamp_folder, lines)
                for _ in lines:
                    if not self.loading:
                        self.status_updated.emit("Loading stopped.")
                        return
                    processed_lines += 1
                    progress = int((processed_lines / total_lines) * 100)
                    self.progress_updated.emit(progress)
                    time.sleep(0.00005)

            if self.loading:
                self.progress_updated.emit(100)
                self.status_updated.emit("Data Loaded Successfully!")
                self.data_loaded.emit(self.data_path)

        finally:
            self.load_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.loading = False


    def stop_loading(self):
        """Stop the data loading process."""
        self.loading = False

    def convert_csv_to_images(self, csv_file, image_folder=None, lines=None):
        import numpy as np
        import cv2
        import os
        import time

        # Default mapping
        label_mapping_train = {
            27: 26,
            26: 35,
            28: 33,
            31: 34,
            30: 27,
            29: 32,
            35: 31,
            32:30,
            33:29,
            34:28
        }

        # Special mapping for test CSV
        label_mapping_test = {
            26: 35,
            27: 26,
            28: 33
        }

        # Select appropriate label mapping
        if "sign_mnist_alpha_digits_test.csv" in csv_file:
            label_mapping = label_mapping_test
        else:
            label_mapping = label_mapping_train

        # If no output folder was provided, create a new one
        if image_folder is None:
            image_folder = os.path.join(os.path.dirname(csv_file), "images_" + str(int(time.time())))
            os.makedirs(image_folder, exist_ok=True)
            for i in range(36):
                os.makedirs(os.path.join(image_folder, str(i)), exist_ok=True)

        # Temporary storage for special remapped classes
        temp_storage = {i: [] for i in label_mapping.keys()}

        # Read lines from file if not already provided
        if lines is None:
            with open(csv_file, 'r') as file:
                lines = file.readlines()[1:]  # Skip header

        # Process each line to create images
        for idx, line in enumerate(lines):
            if not self.loading:
                break  # Stop if loading is cancelled

            values = line.strip().split(',')
            label = int(values[0])
            pixels = np.array(values[1:], dtype=np.uint8).reshape(28, 28)
            image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            image_name = f"{label}_{idx}.png"

            # Normal save
            if label not in label_mapping:
                image_path = os.path.join(image_folder, str(label), image_name)
                cv2.imwrite(image_path, image)
            else:
                # Store temporarily for mapped labels
                temp_storage[label].append((image, image_name))

        # Move remapped images into appropriate folders
        for original_label, images in temp_storage.items():
            mapped_label = label_mapping[original_label]
            if mapped_label is None:
                continue  # Skip if intentionally left empty

            mapped_folder = os.path.join(image_folder, str(mapped_label))
            for image, image_name in images:
                image_path = os.path.join(mapped_folder, image_name)
                cv2.imwrite(image_path, image)

        for class_label in [25, 9]:
            class_path = os.path.join(image_folder, str(class_label))
            if os.path.isdir(class_path):
                os.rmdir(class_path)
                print(f"Removed class folder: {class_path}")
                
        return image_folder
