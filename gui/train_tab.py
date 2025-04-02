from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider
)
from PyQt5.QtCore import Qt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch
import os
import time

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

        self.dataset_path = None

        self.setLayout(layout)

    def set_dataset_path(self, path):
        self.dataset_path = path
        print(f"[TrainTab] Dataset path set to: {path}")


    def start_training(self):
        if self.dataset_path is None:
            print("Dataset path not set. Please load data first.")
            return

        # Collect values from UI
        model_name = self.model_dropdown.currentText()
        split_ratio = self.split_slider.value()
        batch_size = self.batch_slider.value()
        epochs = self.epoch_slider.value()

        print(f"Training {model_name} | Split: {split_ratio}% | Batch: {batch_size} | Epochs: {epochs}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Load dataset using ImageFolder
        dataset = ImageFolder(root=self.dataset_path, transform=transform)

        # Split into train and validation sets
        train_len = int(len(dataset) * (split_ratio / 100))
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Get model
        from core.model_utils import get_model
        model = get_model(model_name)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Time: {duration:.2f}s")

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total
            print(f"Validation Accuracy: {val_acc:.2f}%")

        # Save model
        model_name_file = f"{model_name}_E{epochs}_B{batch_size}_S{split_ratio}.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("models", model_name_file))
        print(f"Model saved as {model_name_file}")