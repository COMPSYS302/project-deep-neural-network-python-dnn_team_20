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
        self.uploaded_csv_files = set()
        super().__init__()
        
        layout = QVBoxLayout()

        self.title_box = QLabel("LOAD")
        self.title_box.setObjectName("title_box")
        self.title_box.setAlignment(Qt.AlignCenter)  # Center the text inside the box
        self.title_box.setWordWrap(True)  # Allow text to wrap

        self.title_box.setFixedHeight(50)  # Set height of the title box
        self.title_box.setFixedWidth(200)  # Set width of the title box

        # Add the title label to the layout
        layout.addWidget(self.title_box, alignment= Qt.AlignTop | Qt.AlignHCenter)

        # Progress bar to show loading progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Button to load data (CSV file)
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)  # Connect to load_data method
        layout.addWidget(self.load_button)

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
        self.loading = True 
        """Method to open a file dialog and select a CSV file to load data."""
        # Open file dialog to select CSV file
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)")

        
        if file_paths:
            self.load_button.setEnabled(False)  # Disable the load button
            self.stop_button.setEnabled(True)  # Enable the stop button
            self.status_updated.emit("Loading data...")  # Update status message

            # Start the data processing in a separate thread
            self.thread = threading.Thread(target=self.process_data, args=(file_paths,))
            self.thread.start()


    def process_data(self, file_paths):
        try:
            # Create output image folder with timestamp
            timestamp_folder = os.path.join(os.path.dirname(file_paths[0]), "images_" + str(int(time.time())))
            os.makedirs(timestamp_folder, exist_ok=True)
            self.data_path = timestamp_folder

            total_lines = 0
            all_lines = []

            # Count total lines (for progress bar)
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    lines = file.readlines()[1:]  # Skip header
                    all_lines.append((file_path, lines))
                    total_lines += len(lines)

            processed_lines = 0
            start_time = time.time()

            for file_path, lines in all_lines:
                self.uploaded_csv_files.add(os.path.basename(file_path))
                self.convert_csv_to_images(file_path, timestamp_folder, lines)

                for _ in lines:
                    if not self.loading:
                        self.status_updated.emit("Loading stopped.")
                        return
                    processed_lines += 1
                    progress = int((processed_lines / total_lines) * 100)
                    self.progress_updated.emit(progress)

                    # Time estimation
                    elapsed_time = time.time() - start_time
                    avg_time_per_line = elapsed_time / processed_lines
                    remaining_time = avg_time_per_line * (total_lines - processed_lines)
                    mins, secs = divmod(int(remaining_time), 60)
                    time_left_str = f"{mins} Min, {secs} Sec left"
                    self.status_updated.emit(f"Loading... {progress}% | Est. time left: {time_left_str}")

                    time.sleep(0.00005)  # Simulate processing delay

            uploaded = self.uploaded_csv_files

            if "sign_mnist_alpha_digits_train.csv" in uploaded and "sign_mnist_alpha_digits_test.csv" in uploaded:
                folders_to_remove = [9, 25]
            elif "sign_mnist_alpha_digits_train.csv" in uploaded:
                folders_to_remove = [9, 25, 26, 35]
            else:
                folders_to_remove = []

            # Remove empty or excluded folders using zero-padded names
            # for i in range(36):
            #     folder_path = os.path.join(self.data_path, f"{i:02d}")
            #     if os.path.isdir(folder_path):
            #         should_remove = (i in folders_to_remove) or (not os.listdir(folder_path))
            #         if should_remove:
            #             try:
            #                 os.rmdir(folder_path)
            #                 print(f"Removed folder: {folder_path}")
            #             except OSError:
            #                 print(f"Could not remove folder (not empty?): {folder_path}")

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

        # Default mapping for train CSV
        label_mapping_train = {
            27: 26,
            26: 35,
            28: 33,
            31: 34,
            30: 27,
            29: 32,
            35: 31,
            32: 30,
            33: 29,
            34: 28
        }

        # Special mapping for test CSV
        label_mapping_test = {
            26: 35,
            27: 26,
            28: 33
        }

        # Select appropriate mapping
        if "sign_mnist_alpha_digits_test.csv" in csv_file:
            label_mapping = label_mapping_test
        else:
            label_mapping = label_mapping_train

        # Create output folder if not provided
        if image_folder is None:
            image_folder = os.path.join(os.path.dirname(csv_file), "images_" + str(int(time.time())))
        
        for i in range(36):
            folder_path = os.path.join(image_folder, f"{i:02d}")
            os.makedirs(folder_path, exist_ok=True)

        # Read CSV lines if not passed in
        if lines is None:
            with open(csv_file, 'r') as file:
                lines = file.readlines()[1:]  # Skip header

        # Process each line
        for idx, line in enumerate(lines):
            if not self.loading:
                break  # Stop if cancelled

            values = line.strip().split(',')
            label = int(values[0])
            pixels = np.array(values[1:], dtype=np.uint8).reshape(28, 28)
            image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            image_name = f"{label}_{idx}.png"

            # Remap label if needed
            final_label = label_mapping.get(label, label)
            folder_name = f"{final_label:02d}"  # Zero-padded
            folder_path = os.path.join(image_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            image_path = os.path.join(folder_path, image_name)
            cv2.imwrite(image_path, image)
            # print(f"[Saved] Label {label} → {final_label} at {image_path}")

        # Add dummy image to empty folders to keep ImageFolder happy
        dummy_image = np.zeros((28, 28, 3), dtype=np.uint8)
        dummy_path = os.path.join(image_folder, "DUMMY.png")

        for i in range(36):
            folder_path = os.path.join(image_folder, f"{i:02d}")
            if not os.listdir(folder_path):  # Folder is empty
                dummy_img_path = os.path.join(folder_path, "dummy.png")
                cv2.imwrite(dummy_img_path, dummy_image)
                print(f"[INFO] Added dummy image to empty folder: {folder_path}")

        return image_folder

