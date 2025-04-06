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
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        
        if file_path:
            self.load_button.setEnabled(False)  # Disable the load button
            self.stop_button.setEnabled(True)  # Enable the stop button
            self.status_updated.emit("Loading data...")  # Update status message
            self.loading = True  # Set loading to True

            # Start the data processing in a separate thread
            self.thread = threading.Thread(target=self.process_data, args=(file_path,))
            self.thread.start()

    def process_data(self, file_path):
        """Method to process the CSV file and convert the data into images."""
        try:
            # Convert CSV data into images and store the image folder path
            image_folder = self.convert_csv_to_images(file_path)
            self.data_path = image_folder

            # Open the CSV file and read all lines
            with open(file_path, 'r') as file:
                lines = file.readlines()
            total = len(lines)  # Total number of lines in the CSV file

            # Process each line in the CSV file (excluding the header)
            for i, line in enumerate(lines):
                if not self.loading:
                    self.status_updated.emit("Loading stopped.")  # If loading is stopped, update status
                    break
                
                # Calculate and emit the progress as a percentage
                # self.progress_updated.emit(int((i / total) * 100))
                progress = int((i / total) * 100)
                self.progress_updated.emit(progress)
                time.sleep(0.00005)  # Small delay to simulate processing time
                # time.sleep(0.05)  # Small delay to show progress update

            # If loading is not stopped, update the status to show success
            if self.loading:
                self.progress_updated.emit(100)  # Set progress to 100%
                self.status_updated.emit("Data Loaded Successfully!")
                self.data_loaded.emit(self.data_path)  # Emit signal with the image folder path
        finally:
            # Enable load button and disable stop button after loading is done or stopped
            self.load_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.loading = False

    def stop_loading(self):
        """Stop the data loading process."""
        self.loading = False

    def convert_csv_to_images(self, csv_file):
        # Default mapping
        label_mapping_train = {
            27: 26,    # Class 26 should contain class 27's images
            26: 35,     # Class 35 should contain class 0's images
            28: 33,    # Class 33 should contain class 2's images
            31 :34,
            30 :27,
            29 :32,
            35: 31
        }

        # Special mapping for "sign_mnist_alpha_digits_test.csv"
        label_mapping_test = {
            34: 28,    
            35: 31,      
            32:30,
        }

        # Determine which mapping to use
        if "sign_mnist_alpha_digits_test.csv" in csv_file:
            label_mapping = label_mapping_test
        else:
            label_mapping = label_mapping_train

        # Create an output folder
        image_folder = os.path.join(os.path.dirname(csv_file), "images_" + str(int(time.time())))
        os.makedirs(image_folder, exist_ok=True)

        # Create subfolders for each label (0-35)
        for i in range(36):
            os.makedirs(os.path.join(image_folder, str(i)), exist_ok=True)

        # Temporary storage for images of classes 26-35
        temp_storage = {i: [] for i in label_mapping.keys()}

        # Read the CSV file and convert each row into an image
        with open(csv_file, 'r') as file:
            lines = file.readlines()
            _ = lines[0].strip().split(',')  # Ignore the header row

            for idx, line in enumerate(lines[1:]):
                if not self.loading:
                    break  # Stop if loading was canceled

                values = line.strip().split(',')
                label = int(values[0])  # First value is the label
                pixels = np.array(values[1:], dtype=np.uint8).reshape(28, 28)

                image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)  
                image_name = f"{label}_{idx}.png"

                # Save images normally if the label is not in the mapping
                if label not in label_mapping:
                    image_path = os.path.join(image_folder, str(label), image_name)
                    cv2.imwrite(image_path, image)
                else:
                    # Store images for mapped classes in temp storage
                    temp_storage[label].append((image, image_name))

        # Move stored images from temp_storage to mapped folders
        for original_label, images in temp_storage.items():
            mapped_label = label_mapping[original_label]

            # Skip if the mapped label is None (class should be empty)
            if mapped_label is None:
                continue

            mapped_folder = os.path.join(image_folder, str(mapped_label))

            for image, image_name in images:
                image_path = os.path.join(mapped_folder, image_name)
                cv2.imwrite(image_path, image)

        return image_folder
