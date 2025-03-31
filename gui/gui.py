import sys
import os
import time
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton,
    QLabel, QProgressBar, QFileDialog, QScrollArea, QGridLayout, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 800, 600)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.load_data_tab = LoadDataTab()
        self.view_data_tab = ViewDataTab()
        
        self.tabs.addTab(self.load_data_tab, "Load Data")
        self.tabs.addTab(self.view_data_tab, "View Data")
        
        self.load_data_tab.data_loaded.connect(self.view_data_tab.load_images)

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
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        for i in range(36):
            class_folder = os.path.join(image_folder, str(i))
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
        
        with open(csv_file, 'r') as file:
            lines = file.readlines()
            header = lines[0].strip().split(',')
            
            for idx, line in enumerate(lines[1:]):
                values = line.strip().split(',')
                label = int(values[0])
                pixels = np.array(values[1:], dtype=np.uint8).reshape(28, 28)
                image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
                image_path = os.path.join(image_folder, str(label), f"{label}_{idx}.png")
                cv2.imwrite(image_path, image)
        
        return image_folder

class ViewDataTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter sign label to filter (A-Z or 0-9)")
        self.filter_input.textChanged.connect(self.update_filter)
        layout.addWidget(self.filter_input)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(600, 500)
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        
        layout.addWidget(self.scroll_area)
        
        self.stats_label = QLabel("Dataset Statistics: ")
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
        
        self.image_folder = None
        self.image_data = []
    
    def load_images(self, folder_path):
        self.image_folder = folder_path
        self.image_data.clear()
        
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)
        
        if not os.path.isdir(folder_path):
            return
        
        for class_label in range(36):
            class_folder = os.path.join(folder_path, str(class_label))
            if os.path.isdir(class_folder):
                for img_file in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_file)
                    self.image_data.append((img_path, str(class_label)))
        
        self.update_statistics()
        self.display_images()
    
    def display_images(self, filter_label=None):
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)
        
        filtered_data = [img for img in self.image_data if filter_label is None or img[1] == filter_label]
        
        for i, (img_path, _) in enumerate(filtered_data):
            pixmap = self.load_pixmap(img_path)
            if pixmap:
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                self.scroll_layout.addWidget(img_label, i // 5, i % 5)
    
    def update_filter(self):
        filter_text = self.filter_input.text().strip().lower() 
        
        if filter_text.isdigit():  
            self.display_images(filter_label=filter_text)
        elif filter_text.isalpha() and len(filter_text) == 1:  
            label = ord(filter_text) - ord('a')
            if 0 <= label <= 25:
                self.display_images(filter_label=str(label))
        elif filter_text.isdigit() and 10 <= int(filter_text) <= 35:
            self.display_images(filter_label=filter_text)
        else:
            self.display_images(filter_label=None)  
    
    def update_statistics(self):
        label_counts = {label: 0 for label in range(36)}
        for _, label in self.image_data:
            label_counts[int(label)] += 1
        stats_text = "Dataset Statistics:\n" + "\n".join(f"{k}: {v} images" for k, v in label_counts.items())
        self.stats_label.setText(stats_text)
    
    def load_pixmap(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img).scaled(100, 100)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())