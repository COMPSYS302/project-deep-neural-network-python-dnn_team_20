import os
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QScrollArea, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap, QImage

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
