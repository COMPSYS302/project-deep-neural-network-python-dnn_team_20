from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea, QListWidget, QListWidgetItem, QHBoxLayout, QAbstractItemView, QGridLayout, QCheckBox, QFrame
"""
test_tab.py
This module defines the `TestTab` class, which is a PyQt5-based GUI component for testing deep learning models. 
It provides functionality to load trained models, test them on selected images, use webcam input for predictions, 
and visualize results.
Classes:
    - TestTab: A QWidget subclass that implements the testing interface.
Dependencies:
    - PyQt5: For GUI components and layout management.
    - PyTorch: For loading and running deep learning models.
    - torchvision: For image transformations and dataset handling.
    - matplotlib: For plotting prediction charts.
    - OpenCV: For webcam integration and image processing.
    - PIL (Pillow): For image manipulation.
    - core.model_utils: For loading model architectures.
Class: TestTab
--------------
Attributes:
    - train_tab (TrainTab): Reference to the training tab for shared resources like model dropdown and device.
    - model (torch.nn.Module): The loaded PyTorch model for testing.
    - selected_image_paths (list): List of file paths for selected images.
    - image_checkboxes (list): List of checkboxes associated with validation images.
    - device (torch.device): The device (CPU/GPU) on which the model is loaded.
Methods:
    - __init__(train_tab, parent=None): Initializes the TestTab GUI layout and connects signals to slots.
    - load_model_from_file(): Opens a file dialog to load a trained PyTorch model and updates the UI.
    - open_validation_viewer(): Displays validation images for selection and testing.
    - hide_validation_area(): Hides the validation image viewer area.
    - show_validation_area(): Shows the validation image viewer area.
    - predict_selected_images(): Predicts labels for selected validation images and displays results.
    - reset_image_selection(): Resets the selection of validation images.
    - test_on_selected_images(): Tests the model on user-selected images and displays predictions.
    - test_with_webcam(): Captures webcam input, predicts the label, and displays results.
    - hide_prediction_display(): Hides the webcam image and prediction chart.
    - show_prediction_display(): Shows the webcam image and prediction chart.
    - map_predicted_to_char(predicted): Maps a predicted index to its corresponding character or label.
Notes:
    - The `load_model_from_file` method infers the model architecture from the filename or a dropdown in the training tab.
    - The `open_validation_viewer` method supports both validation datasets from the training tab and user-selected folders.
    - The `test_with_webcam` method allows real-time predictions using webcam input.
    - The `test_on_selected_images` method visualizes the top-5 predictions using a bar chart.
    - The `map_predicted_to_char` method maps indices to characters (A-Z, 0-9) or special labels.
Inputs:
    - Trained PyTorch model files (.pt).
    - Validation images (from dataset or user-selected folder).
    - Webcam input for real-time testing.
Outputs:
    - Predicted labels and probabilities displayed in the GUI.
    - Accuracy and detailed results for selected images.
    - Bar chart visualization of top-5 predictions.
Usage:
    - Integrate this class into a PyQt5 application as a tab for testing deep learning models.
    - Ensure the `core.model_utils.get_model` function is implemented to return the correct model architecture.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import os
import cv2
from core.model_utils import get_model  # make sure this works for your models
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class TestTab(QWidget):
    def __init__(self, train_tab, parent=None):
        super().__init__(parent)
        self.train_tab = train_tab  # Reference to TrainTab
        self.model = None  # Placeholder for the loaded model

        # Main layout for the TestTab
        layout = QVBoxLayout()

        # Title box at the top of the tab
        self.title_box = QLabel("TEST")
        self.title_box.setObjectName("title_box")  # Set object name for styling
        self.title_box.setAlignment(Qt.AlignCenter)  # Center the text inside the box
        self.title_box.setWordWrap(True)  # Allow text to wrap
        self.title_box.setFixedHeight(50)  # Set height of the title box
        self.title_box.setFixedWidth(200)  # Set width of the title box

        # Frame for displaying prediction charts
        self.prediction_chart_frame = QFrame()
        self.prediction_chart_layout = QVBoxLayout(self.prediction_chart_frame)

        # Set up the matplotlib figure and canvas for prediction visualization
        self.figure = plt.Figure(figsize=(4, 2))
        self.canvas = FigureCanvas(self.figure)
        self.prediction_chart_layout.addWidget(self.canvas)

        # Add title box to the layout
        layout.addWidget(self.title_box, alignment=Qt.AlignTop | Qt.AlignHCenter)

        # Button to load a trained model
        self.load_model_btn = QPushButton("Load Trained Model From File")
        # Button to test the model with selected images
        self.test_images_btn = QPushButton("Test with Images")
        # Label to display test results
        self.result_label = QLabel("Test results will appear here.")
        # Button to test the model using webcam input
        self.webcam_button = QPushButton("Test with Webcam")
        # Label to display webcam images
        self.webcam_image_label = QLabel()
        self.webcam_image_label.setAlignment(Qt.AlignCenter)  # Center the webcam image
        # Button to open the validation image viewer
        self.view_val_images_btn = QPushButton("Open Validation Image Viewer")
        # Button to predict labels for selected images
        self.predict_selected_btn = QPushButton("Predict Selected Images")
        # Button to reset the selection of validation images
        self.reset_selection_btn = QPushButton("Reset Selection")
        self.reset_selection_btn.hide()  # Initially hidden

        # Scroll area for displaying validation images
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_grid_widget = QWidget()
        self.image_grid_layout = QGridLayout()
        self.image_grid_widget.setLayout(self.image_grid_layout)
        self.image_scroll_area.setWidget(self.image_grid_widget)
        self.image_scroll_area.setFixedHeight(250)  # Limit the height of the scroll area
        self.image_scroll_area.hide()  # Initially hidden

        # Add widgets to the main layout
        layout.addWidget(self.load_model_btn)
        layout.addWidget(self.webcam_button)
        layout.addWidget(self.test_images_btn)
        layout.addWidget(self.webcam_image_label)
        layout.addWidget(self.prediction_chart_frame)
        layout.addWidget(self.result_label)
        layout.addWidget(self.view_val_images_btn)
        layout.addWidget(self.reset_selection_btn)
        layout.addWidget(self.image_scroll_area)
        layout.addWidget(self.predict_selected_btn)

        # Set the main layout for the TestTab
        self.setLayout(layout)

        # Connect buttons to their respective methods
        self.load_model_btn.clicked.connect(self.load_model_from_file)
        self.test_images_btn.clicked.connect(self.test_on_selected_images)
        self.webcam_button.clicked.connect(self.test_with_webcam)
        self.view_val_images_btn.clicked.connect(self.open_validation_viewer)
        self.predict_selected_btn.clicked.connect(self.predict_selected_images)
        self.reset_selection_btn.clicked.connect(self.reset_image_selection)

        # Initialize attributes for image selection and checkboxes
        self.model = None  # Placeholder for the loaded model
        self.predict_selected_btn.hide()  # Hide the predict button initially
        self.selected_image_paths = []  # List to store paths of selected images
        self.image_checkboxes = []  # List to store checkboxes for images

    def load_model_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "models", "PyTorch Model (*.pt)")
        if not file_path:
            return  # User cancelled

        try:
            # Infer model type from filename
            filename = os.path.basename(file_path).lower()
            if "alexnet" in filename:
                model_name = "AlexNet"
            elif "inceptionv3" in filename or "inception" in filename:
                model_name = "InceptionV3"
            elif "sesame" in filename:
                model_name = "Sesame 1.0"
            else:
                # If model name can't be inferred, fall back to dropdown if available
                model_name = getattr(self.train_tab, 'model_dropdown', None)
                model_name = model_name.currentText() if model_name else "AlexNet"  # Safe fallback

            # Update dropdown UI to reflect loaded model (optional UX consistency)
            if hasattr(self.train_tab, 'model_dropdown'):
                index = self.train_tab.model_dropdown.findText(model_name)
                if index != -1:
                    self.train_tab.model_dropdown.setCurrentIndex(index)

            # Get model architecture safely
            model = get_model(model_name)

            # Ensure device is set even if no training has occurred
            if not hasattr(self.train_tab, 'device'):
                self.train_tab.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = self.train_tab.device

            # Try loading model weights
            try:
                model.load_state_dict(torch.load(file_path, map_location=device))
            except RuntimeError as e:
                self.result_label.setText("Model architecture mismatch. Please ensure you're loading the correct model file.")
                return

            model.to(device)
            model.eval()

            # Set attributes
            self.model = model
            self.device = device
            self.result_label.setText(f"Loaded model: {os.path.basename(file_path)} ({model_name})")

        except Exception as e:
            self.result_label.setText(f"Failed to load model:\n{str(e)}")

            # Method to open the validation image viewer
            def open_validation_viewer(self):
                import torchvision.transforms.functional as TF
                from torchvision.datasets import ImageFolder
                from torchvision import transforms
                from torch.utils.data import DataLoader

                # Hide prediction display and show validation area
                self.hide_prediction_display()
                self.show_validation_area()

                # Check if validation dataset is available from the training tab
                val_dataset = getattr(self.train_tab, 'val_dataset', None)
                if val_dataset:
                    print("using val dataset from train")
                    dataset_source = "train_tab"

                    # Map class indices to class names if available
                    if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'class_to_idx'):
                        self.class_to_idx = val_dataset.dataset.class_to_idx
                        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

                # If no validation dataset is available, prompt the user to select a folder
                if val_dataset is None:
                    folder = QFileDialog.getExistingDirectory(self, "Select Test Image Folder", self.train_tab.dataset_path or "")
                    if not folder:
                        self.result_label.setText("No validation dataset or folder selected.")
                        return

                    # Determine image size and transformations based on the model
                    model_name = self.train_tab.model_dropdown.currentText()
                    img_size = (224, 224) if model_name in ["AlexNet", "InceptionV3"] else (28, 28)
                    transform = transforms.Compose([
                        transforms.Grayscale() if model_name == "Sesame 1.0" else transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)) if model_name == "Sesame 1.0"
                            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

                # Show the scroll area and prediction button
                self.image_scroll_area.show()
                self.predict_selected_btn.show()
                self.image_grid_layout.setSpacing(10)

                # Clear existing widgets in the grid layout
                for i in reversed(range(self.image_grid_layout.count())):
                    widget_to_remove = self.image_grid_layout.itemAt(i).widget()
                    self.image_grid_layout.removeWidget(widget_to_remove)
                    widget_to_remove.setParent(None)

                # Clear previous image checkboxes and selections
                self.image_checkboxes.clear()
                self.selected_image_paths.clear()

                # Populate the grid layout with images and checkboxes
                row, col = 0, 0
                for idx in range(min(len(val_dataset), 100)):  # Limit to 100 images for performance
                    image_tensor, label = val_dataset[idx]
                    try:
                        # Convert image tensor to PIL image
                        image = TF.to_pil_image(image_tensor.cpu())
                    except Exception as e:
                        print(f"Error converting image {idx}: {e}")
                        continue

                    # Convert PIL image to Qt image
                    qimage = QImage(image.convert("RGB").tobytes(), image.width, image.height, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage).scaled(64, 64, Qt.KeepAspectRatio)

                    # Create a checkbox and label for the image
                    checkbox = QCheckBox()
                    image_label = QLabel()
                    image_label.setPixmap(pixmap)
                    image_label.setFixedSize(70, 70)
                    image_label.setAlignment(Qt.AlignCenter)

                    # Create a frame to hold the image and checkbox
                    frame = QFrame()
                    frame_layout = QVBoxLayout()
                    frame_layout.addWidget(image_label)
                    frame_layout.addWidget(checkbox)
                    frame.setLayout(frame_layout)

                    # Add the frame to the grid layout
                    self.image_grid_layout.addWidget(frame, row, col)
                    self.image_checkboxes.append((checkbox, image_tensor, label))

                    # Update row and column for the next image
                    col += 1
                    if col >= 6:
                        col = 0
                        row += 1

                # Show the reset selection button
                self.reset_selection_btn.show()

                # Update the result label to indicate the source of validation images
                self.result_label.setText(f"Loaded validation images from {'training split' if dataset_source == 'train_tab' else 'folder'} ✅")

            # Method to hide the validation image viewer area
            def hide_validation_area(self):
                self.image_scroll_area.hide()
                self.predict_selected_btn.hide()
                self.reset_selection_btn.hide()

            # Method to show the validation image viewer area
            def show_validation_area(self):
                self.image_scroll_area.show()
                self.predict_selected_btn.show()
                self.reset_selection_btn.show()

            # Method to predict labels for selected validation images
            def predict_selected_images(self):
                if self.model is None:
                    self.result_label.setText("No model loaded.")
                    return

                # Get the device and model name
                device = getattr(self.train_tab, 'device', torch.device('cpu'))
                model_name = self.train_tab.model_dropdown.currentText()
                self.model.eval()

                correct = total = 0
                results = []

                # Perform predictions on selected images
                with torch.no_grad():
                    for checkbox, img_tensor, true_label in self.image_checkboxes:
                        if not checkbox.isChecked():
                            continue

                        # Prepare the input tensor and perform inference
                        input_tensor = img_tensor.unsqueeze(0).to(device)
                        output = self.model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        predicted_idx = predicted.item()
                        predicted_char = self.map_predicted_to_char(predicted_idx)

                        # Map the true label to its corresponding character
                        try:
                            if hasattr(self, 'idx_to_class'):
                                true_index = int(self.idx_to_class[true_label])
                            else:
                                true_index = true_label
                            true_char = self.map_predicted_to_char(true_index)
                        except Exception as e:
                            print(f"[Warning] Could not map true label: {e}")
                            true_char = str(true_label)

                        # Check if the prediction is correct
                        is_correct = predicted_idx == true_index
                        correct += int(is_correct)
                        total += 1

                        # Append the result to the results list
                        results.append(
                            f"True: {true_char}, Predicted: {predicted_char} {'✔️' if is_correct else '❌'}"
                        )

                # Calculate accuracy and update the result label
                accuracy = f"\nAccuracy: {100.0 * correct / total:.2f}%" if total > 0 else ""
                self.result_label.setText("Results:\n" + "\n".join(results) + accuracy)

            # Method to reset the selection of validation images
            def reset_image_selection(self):
                self.result_label.setText("Results cleared.")
                for checkbox, _, _ in self.image_checkboxes:
                    checkbox.setChecked(False)

            # Method to test the model on user-selected images
            def test_on_selected_images(self):
                from PIL import Image
                import numpy as np
                import matplotlib.pyplot as plt
                import torch.nn.functional as F
                from torchvision import transforms

                # Hide validation area and show prediction display
                self.hide_validation_area()
                self.show_prediction_display()

                # Open file dialog to select test images
                file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Test Images", "", "Images (*.png *.jpg *.jpeg)")
                if not file_paths or self.model is None:
                    self.result_label.setText("No model loaded or no images selected.")
                    return

                # Determine image size and transformations based on the model
                model_name = self.train_tab.model_dropdown.currentText()
                img_size = (224, 224) if model_name in ["AlexNet", "InceptionV3"] else (28, 28)
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                # Get the device and set the model to evaluation mode
                device = getattr(self.train_tab, 'device', torch.device('cpu'))
                self.model.eval()
                results = []

                # Clear previous image and graph
                self.webcam_image_label.clear()
                self.figure.clear()

                # Perform predictions on the selected images
                with torch.no_grad():
                    for path in file_paths:
                        img = cv2.imread(path)
                        if img is None:
                            results.append(f"Failed to load image: {os.path.basename(path)}")
                            continue

                        # Convert the image to RGB and apply transformations
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        img_tensor = transform(img_rgb).unsqueeze(0).to(device)

                        # Perform inference and get top-5 predictions
                        output = self.model(img_tensor)
                        probs = F.softmax(output, dim=1)[0]
                        top_probs, top_indices = torch.topk(probs, 5)

                        # Map the predicted index to its corresponding character
                        predicted_index = top_indices[0].item()
                        predicted_char = self.map_predicted_to_char(predicted_index)

                        # Try to map the true label if available
                        try:
                            folder_name = os.path.basename(os.path.dirname(path))
                            true_index = int(folder_name)
                            true_char = self.map_predicted_to_char(true_index)
                            results.append(
                                f"{os.path.basename(path)} - True: {true_index} ({true_char}), "
                                f"Predicted: {predicted_index} ({predicted_char})"
                            )
                        except ValueError:
                            results.append(
                                f"{os.path.basename(path)} - Predicted: {predicted_index} ({predicted_char})"
                            )

                        # Display the image in the GUI
                        display_img = pil_img.resize((128, 128))
                        qimage = QImage(display_img.convert("RGB").tobytes(), display_img.width, display_img.height, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimage)
                        self.webcam_image_label.setPixmap(pixmap)
                        self.webcam_image_label.setAlignment(Qt.AlignCenter)
                        self.webcam_image_label.setFixedSize(140, 140)

                        # Draw the prediction chart
                        ax = self.figure.add_subplot(111)
                        labels = [self.map_predicted_to_char(i.item()) for i in top_indices]
                        values = [p.item() * 100 for p in top_probs]

                        ax.bar(labels, values, color='skyblue')
                        ax.set_title("Top 5 Predictions")
                        ax.set_ylabel("Probability (%)")
                        ax.set_xlabel("Class")
                        ax.set_ylim([0, 100])
                        self.canvas.draw()

                        break  # Process only the first image for now

                # Update the result label with the predictions
                self.result_label.setText("Results:\n" + "\n".join(results))

            # Method to map a predicted index to its corresponding character or label
            def map_predicted_to_char(self, predicted):
                if 0 <= predicted <= 25:
                    return chr(ord('A') + predicted)  # A-Z
                elif 26 <= predicted <= 35:
                    return str(predicted - 26)        # 0-9
                elif predicted == 37:
                    return "?"  # For special/untrained classes
                return str(predicted)  # Fallback for anything unexpected

            # Method to test the model using webcam input
            def test_with_webcam(self):
                from PIL import Image
                import datetime
                import torch.nn.functional as F

                # Hide validation area and show prediction display
                self.hide_validation_area()
                self.show_prediction_display()

                if self.model is None:
                    self.result_label.setText("No model loaded.")
                    return

                # Get the device and model name
                device = getattr(self.train_tab, 'device', torch.device('cpu'))
                model_name = self.train_tab.model_dropdown.currentText()

                # Determine image size and transformations based on the model
                is_rgb_model = model_name in ["AlexNet", "InceptionV3", "Sesame 1.0"]
                if model_name == "Sesame 1.0":
                    img_size = (28, 28)
                    to_color = transforms.Lambda(lambda x: x.convert("RGB"))
                    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                else:
                    img_size = (224, 224)
                    to_color = transforms.Lambda(lambda x: x.convert("RGB"))
                    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                qimage_format = QImage.Format_RGB888 if is_rgb_model else QImage.Format_Grayscale8
                transform = transforms.Compose([
                    to_color,
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    normalize
                ])

                # Open the webcam
                self.model.eval()
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    self.result_label.setText("Webcam not accessible.")
                    return

                self.result_label.setText("Press 'q' to quit webcam, 'c' to capture and test.")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Display the webcam feed
                    cv2.imshow("Press 'c' to capture", frame)
                    key = cv2.waitKey(1)

                    if key == ord('q'):  # Quit the webcam
                        break
                    elif key == ord('c'):  # Capture and test the frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        transformed_tensor = transform(pil_img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = self.model(transformed_tensor)
                            probs = F.softmax(output, dim=1)[0]
                            top_probs, top_indices = torch.topk(probs, 5)
                            predicted_char = self.map_predicted_to_char(top_indices[0].item())

                        # Update the result label with the prediction
                        self.result_label.setText(f"Webcam Prediction: {predicted_char}")

                        # Display the captured image in the GUI
                        display_img = pil_img.resize((128, 128))
                        qimage = QImage(display_img.convert("RGB").tobytes(), display_img.width, display_img.height, qimage_format)
                        pixmap = QPixmap.fromImage(qimage)
                        self.webcam_image_label.setPixmap(pixmap)
                        self.webcam_image_label.setAlignment(Qt.AlignCenter)
                        self.webcam_image_label.setFixedSize(140, 140)

                        # Plot the top-5 predictions
                        self.figure.clear()
                        ax = self.figure.add_subplot(111)
                        labels = [self.map_predicted_to_char(i.item()) for i in top_indices]
                        values = [p.item() * 100 for p in top_probs]

                        ax.bar(labels, values)
                        ax.set_title("Top 5 Predictions")
                        ax.set_ylabel("Probability (%)")
                        ax.set_xlabel("Class")
                        self.canvas.draw()

                        break

                # Release the webcam and close the window
                cap.release()
                cv2.destroyAllWindows()

            # Method to hide the prediction display
            def hide_prediction_display(self):
                self.webcam_image_label.clear()
                self.webcam_image_label.hide()
                self.prediction_chart_frame.hide()

            # Method to show the prediction display
            def show_prediction_display(self):
                self.webcam_image_label.show()
                self.prediction_chart_frame.show()
