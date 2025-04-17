from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from torch.utils.data import DataLoader
import torch
import os
from core.model_utils import get_model  # make sure this works for your models

class TestTab(QWidget):
    def __init__(self, train_tab, parent=None):
        super().__init__(parent)
        self.train_tab = train_tab  # Reference to TrainTab

        layout = QVBoxLayout()

        self.load_model_btn = QPushButton("Load Trained Model From File")
        self.test_memory_model_btn = QPushButton("Test Current Trained Model")
        self.result_label = QLabel("Test results will appear here.")

        layout.addWidget(self.load_model_btn)
        layout.addWidget(self.test_memory_model_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.load_model_btn.clicked.connect(self.load_model_from_file)
        self.test_memory_model_btn.clicked.connect(self.test_model_in_memory)

        self.model = None

    def load_model_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "models", "PyTorch Model (*.pt)")
        if file_path:
            model_name = self.train_tab.model_dropdown.currentText()
            model = get_model(model_name)
            model.load_state_dict(torch.load(file_path, map_location=self.train_tab.device))
            model.to(self.train_tab.device)
            model.eval()
            self.model = model
            self.result_label.setText(f"Loaded model: {os.path.basename(file_path)}")

    def test_model_in_memory(self):
        model = self.model if self.model else getattr(self.train_tab, 'trained_model', None)
        val_dataset = getattr(self.train_tab, 'val_dataset', None)
        device = getattr(self.train_tab, 'device', torch.device('cpu'))

        if model is None or val_dataset is None:
            self.result_label.setText("Model or validation dataset not available.")
            return

        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        self.result_label.setText(f"Validation Accuracy: {accuracy:.2f}%")
