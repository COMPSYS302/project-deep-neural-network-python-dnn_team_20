from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea, QListWidget, QListWidgetItem, QHBoxLayout, QAbstractItemView, QGridLayout, QCheckBox, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
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

        self.title_box = QLabel("TEST")
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
        self.webcam_image_label = QLabel()
        self.view_val_images_btn = QPushButton("Open Validation Image Viewer")
        self.predict_selected_btn = QPushButton("Predict Selected Images")
        self.reset_selection_btn = QPushButton("Reset Selection")

        self.reset_selection_btn.hide()

        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_grid_widget = QWidget()
        self.image_grid_layout = QGridLayout()
        self.image_grid_widget.setLayout(self.image_grid_layout)
        self.image_scroll_area.setWidget(self.image_grid_widget)
        self.image_scroll_area.setFixedHeight(300)
        self.image_scroll_area.hide()
      
        layout.addWidget(self.load_model_btn)     
        layout.addWidget(self.webcam_button)
        layout.addWidget(self.test_images_btn)
        layout.addWidget(self.webcam_image_label)
        layout.addWidget(self.test_memory_model_btn)
        layout.addWidget(self.result_label)
        layout.addWidget(self.view_val_images_btn)
        layout.addWidget(self.reset_selection_btn)
        layout.addWidget(self.image_scroll_area)
        layout.addWidget(self.predict_selected_btn)
        self.setLayout(layout)

        self.load_model_btn.clicked.connect(self.load_model_from_file)
        self.test_memory_model_btn.clicked.connect(self.test_model_in_memory)
        self.test_images_btn.clicked.connect(self.test_on_selected_images)
        self.webcam_button.clicked.connect(self.test_with_webcam)
        self.view_val_images_btn.clicked.connect(self.open_validation_viewer)
        self.predict_selected_btn.clicked.connect(self.predict_selected_images)
        self.reset_selection_btn.clicked.connect(self.reset_image_selection)
        self.model = None

       
        self.predict_selected_btn.hide()

        self.selected_image_paths = []
        self.image_checkboxes = []

    def load_model_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "models", "PyTorch Model (*.pt)")
        if file_path:
            model_name = self.train_tab.model_dropdown.currentText()
            model = get_model(model_name)
            device = getattr(self.train_tab, 'device', torch.device('cpu'))
            model.load_state_dict(torch.load(file_path, map_location=device))
            model.to(device)
            self.device = device  
            model.eval()
            self.model = model
            self.result_label.setText(f"Loaded model: {os.path.basename(file_path)}")


    def open_validation_viewer(self):
        import torchvision.transforms.functional as TF
        from torchvision.datasets import ImageFolder
        from torchvision import transforms
        from torch.utils.data import DataLoader

        val_dataset = getattr(self.train_tab, 'val_dataset', None)
        if val_dataset:
            print("using val dataset from train")
            dataset_source = "train_tab"

            if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'class_to_idx'):
                self.class_to_idx = val_dataset.dataset.class_to_idx
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}


        # If val_dataset isn't available, ask user to choose a folder
        if val_dataset is None:
            folder = QFileDialog.getExistingDirectory(self, "Select Test Image Folder", self.train_tab.dataset_path or "")
            if not folder:
                self.result_label.setText("No validation dataset or folder selected.")
                return

            model_name = self.train_tab.model_dropdown.currentText()
            img_size = (224, 224) if model_name in ["AlexNet", "InceptionV3"] else (28, 28)

            transform = transforms.Compose([
                transforms.Grayscale() if model_name == "Sesame 1.0" else transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) if model_name == "Sesame 1.0"
                    else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            val_dataset = ImageFolder(root=folder, transform=transform)
            dataset_source = "folder"

        self.image_scroll_area.show()
        self.predict_selected_btn.show()
        self.image_grid_layout.setSpacing(10)

        # Clear existing widgets
        for i in reversed(range(self.image_grid_layout.count())):
            widget_to_remove = self.image_grid_layout.itemAt(i).widget()
            self.image_grid_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        self.image_checkboxes.clear()
        self.selected_image_paths.clear()

        row, col = 0, 0
        for idx in range(min(len(val_dataset), 100)):  # Limit for performance
            image_tensor, label = val_dataset[idx]

            try:
                image = TF.to_pil_image(image_tensor.cpu())
            except Exception as e:
                print(f"Error converting image {idx}: {e}")
                continue

            # Convert to Qt image
            qimage = QImage(image.convert("RGB").tobytes(), image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(64, 64, Qt.KeepAspectRatio)

            checkbox = QCheckBox()
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setFixedSize(70, 70)
            image_label.setAlignment(Qt.AlignCenter)

            # Layout for image + checkbox
            frame = QFrame()
            frame_layout = QVBoxLayout()
            frame_layout.addWidget(image_label)
            frame_layout.addWidget(checkbox)
            frame.setLayout(frame_layout)

            self.image_grid_layout.addWidget(frame, row, col)
            self.image_checkboxes.append((checkbox, image_tensor, label))

            col += 1
            if col >= 6:
                col = 0
                row += 1

        self.reset_selection_btn.show()
        self.result_label.setText(f"Loaded validation images from {'training split' if dataset_source == 'train_tab' else 'folder'} ✅")



    def predict_selected_images(self):
        if self.model is None:
            self.result_label.setText("No model loaded.")
            return

        device = getattr(self.train_tab, 'device', torch.device('cpu'))
        model_name = self.train_tab.model_dropdown.currentText()
        self.model.eval()

        correct = total = 0
        results = []

        with torch.no_grad():
            for checkbox, img_tensor, true_label in self.image_checkboxes:
                if not checkbox.isChecked():
                    continue

                input_tensor = img_tensor.unsqueeze(0).to(device)
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_idx = predicted.item()
                predicted_char = self.map_predicted_to_char(predicted_idx)

                # 🔧 Map the true label using idx_to_class if available
                try:
                    if hasattr(self, 'idx_to_class'):
                        # Handles both Subset(val_dataset) and ImageFolder
                        true_index = int(self.idx_to_class[true_label])
                    else:
                        true_index = true_label

                    true_char = self.map_predicted_to_char(true_index)
                except Exception as e:
                    print(f"[Warning] Could not map true label: {e}")
                    true_char = str(true_label)

                is_correct = predicted_idx == true_index
                correct += int(is_correct)
                total += 1

                results.append(
                    f"True: {true_char}, Predicted: {predicted_char} {'✔️' if is_correct else '❌'}"
                )

        accuracy = f"\nAccuracy: {100.0 * correct / total:.2f}%" if total > 0 else ""
        self.result_label.setText("Results:\n" + "\n".join(results) + accuracy)



    def reset_image_selection(self):
        self.result_label.setText("Results cleared.")
        for checkbox, _, _ in self.image_checkboxes:
            checkbox.setChecked(False)


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
        img_size = (224, 224) if model_name in ["AlexNet", "InceptionV3"] else (28, 28)

        # Handle color channels based on model type
        to_color = transforms.Grayscale() if model_name == "Sesame 1.0" else transforms.Lambda(lambda x: x.convert("RGB"))

        normalize = (
            transforms.Normalize((0.5,), (0.5,)) if model_name == "Sesame 1.0"
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

        transform = transforms.Compose([
            transforms.ToPILImage(),
            to_color,
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])
        device = getattr(self.train_tab, 'device', torch.device('cpu'))

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

                predicted_index = predicted.item()
                predicted_char = self.map_predicted_to_char(predicted_index)

                # Try to get true label from folder name, but handle non-numeric folder names
                try:
                    folder_name = os.path.basename(os.path.dirname(path))
                    true_index = int(folder_name)
                    true_char = self.map_predicted_to_char(true_index)
                    results.append(f"{os.path.basename(path)} - True Class: {true_index} ({true_char}), Predicted Class: {predicted_index} ({predicted_char})")
                except ValueError:
                    # If folder name is not a number, just show prediction
                    results.append(f"{os.path.basename(path)} - Predicted Class: {predicted_index} ({predicted_char})")

        # After loop, set the result label text
        self.result_label.setText("Results:\n" + "\n".join(results))
    
    def map_predicted_to_char(self, predicted):
        if 0 <= predicted <= 25:
            return chr(ord('A') + predicted)  # A-Z
        elif 26 <= predicted <= 35:
            return str(predicted - 26)        # 0-9
        elif predicted == 37:
            return "?"  # For special/untrained classes
        return str(predicted)  # Fallback for anything unexpected

    def test_with_webcam(self):
        import datetime
        from PIL import Image
        import numpy as np

        if self.model is None:
            self.result_label.setText("No model loaded.")
            return

        model_name = self.train_tab.model_dropdown.currentText()

        transform, img_size = self.get_transform_pipeline()


        device = getattr(self.train_tab, 'device', torch.device('cpu'))
        self.model.eval()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.result_label.setText("Webcam not accessible.")
            return

        self.result_label.setText("Press 'q' to quit webcam, 'c' to capture and predict.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Press 'c' to capture", frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('c'):
                # Convert frame to RGB and apply transform
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                transformed_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = self.model(transformed_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_idx = predicted.item()
                    predicted_char = self.map_predicted_to_char(predicted_idx)

                self.result_label.setText(f"Webcam Prediction: {predicted_char} (Class {predicted_idx})")

                # Show image in GUI
                display_img = pil_img.resize((128, 128))  # resize for UI
                qimage = QImage(display_img.convert("RGB").tobytes(), display_img.width, display_img.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.webcam_image_label.setPixmap(pixmap)
                self.webcam_image_label.setAlignment(Qt.AlignCenter)

                break

        cap.release()
        cv2.destroyAllWindows()



    def get_transform_pipeline(self):
        model_name = self.train_tab.model_dropdown.currentText()
        img_size = (224, 224) if model_name in ["AlexNet", "InceptionV3"] else (28, 28)

        to_color = transforms.Grayscale() if model_name == "Sesame 1.0" else transforms.Lambda(lambda x: x.convert("RGB"))

        normalize = (
            transforms.Normalize((0.5,), (0.5,)) if model_name == "Sesame 1.0"
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

        return transforms.Compose([
            to_color,
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ]), img_size

