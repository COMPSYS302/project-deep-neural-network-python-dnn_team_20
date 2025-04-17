from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import Qt
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import os
import cv2
from core.model_utils import get_model  # make sure this works for your models

class TestTab(QWidget):
    def __init__(self, train_tab, parent=None):
        super().__init__(parent)
        self.train_tab = train_tab  # Reference to TrainTab
        self.model = None

        layout = QVBoxLayout()

        self.title_box = QLabel("LOAD")
        self.title_box.setObjectName("title_box")
        self.title_box.setAlignment(Qt.AlignCenter)  # Center the text inside the box
        self.title_box.setWordWrap(True)  # Allow text to wrap

        self.title_box.setFixedHeight(50)  # Set height of the title box
        self.title_box.setFixedWidth(200)  # Set width of the title box

        layout.addWidget(self.title_box, alignment= Qt.AlignTop | Qt.AlignHCenter)

        self.load_model_btn = QPushButton("Load Trained Model From File")
        self.test_memory_model_btn = QPushButton("Test Current Trained Model")
        self.test_images_btn = QPushButton("Test with Images")
        self.result_label = QLabel("Test results will appear here.")
        self.webcam_button = QPushButton("Test with Webcam")
      
        layout.addWidget(self.load_model_btn)     
        layout.addWidget(self.webcam_button)
        layout.addWidget(self.test_images_btn)
        layout.addWidget(self.test_memory_model_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.load_model_btn.clicked.connect(self.load_model_from_file)
        self.test_memory_model_btn.clicked.connect(self.test_model_in_memory)
        self.test_images_btn.clicked.connect(self.test_on_selected_images)
        self.webcam_button.clicked.connect(self.test_with_webcam)

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

    def test_on_selected_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Test Images", "", "Images (*.png *.jpg *.jpeg)")
        if not file_paths or self.model is None:
            self.result_label.setText("No model loaded or no images selected.")
            return

        model_name = self.train_tab.model_dropdown.currentText()
        img_size = (28, 28) if model_name == "Sesame 1.0" else (224, 224)
        to_color = transforms.Grayscale() if model_name == "Sesame 1.0" else transforms.Lambda(lambda x: x.convert('RGB'))
        normalize = transforms.Normalize((0.5,), (0.5,)) if model_name == "Sesame 1.0" else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        transform = transforms.Compose([
            transforms.ToPILImage(),
            to_color,
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])

        device = self.train_tab.device
        self.model.eval()
        results = []

        with torch.no_grad():
            for path in file_paths:
                img = cv2.imread(path)

                if img is None:
                    results.append(f"Failed to load image: {os.path.basename(path)}")
                    continue
                
                img_tensor = transform(img).unsqueeze(0).to(device)
                output = self.model(img_tensor)
                _, predicted = torch.max(output, 1)
                results.append(f"{os.path.basename(path)}: {predicted.item()}")

        self.result_label.setText("Results:\n" + "\n".join(results))

    def test_with_webcam(self):
        if self.model is None:
            self.result_label.setText("No model loaded.")
            return

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        device = self.train_tab.device
        self.model.eval()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.result_label.setText("Webcam not accessible.")
            return

        self.result_label.setText("Press 'q' to quit webcam.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Show webcam feed
            cv2.imshow("Press 'c' to capture", frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('c'):
                img_tensor = transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = self.model(img_tensor)
                    _, predicted = torch.max(output, 1)
                self.result_label.setText(f"Webcam Prediction: {predicted.item()}")

        cap.release()
        cv2.destroyAllWindows()