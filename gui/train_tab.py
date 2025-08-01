import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider, QProgressBar, QApplication)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from torchvision import transforms
from torchvision.datasets import ImageFolder
from core.model_utils import get_model

class TrainerThread(QThread):
    """
    A QThread to run the training loop in the background.
    """
    # Custom signals to send back to the main thread:
    progress_signal = pyqtSignal(int, float, float)  
    #        ^ epoch      ^ train_loss  ^ val_accuracy

    finished_signal = pyqtSignal(str)  # emits model filename when done

    def __init__(self, model, train_loader, val_loader, epochs, device, model_file_path):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.model_file_path = model_file_path
        self._stop_flag = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def stop(self):
        """Call this to request stopping the thread early."""
        self._stop_flag = True

    def run(self):
        """Main training loop."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

        start_time = time.time()
        
        # Training loop
        for epoch in range(self.epochs):
            if self._stop_flag:
                break
            
            # Training
            self.model.train()
            running_loss = 0.0
            num_batches = 0

            # Loop through the training data
            for images, labels in self.train_loader:
                if self._stop_flag:
                    break
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss for the epoch
            avg_loss = running_loss / num_batches
            
            print(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.2f}")

            # Validation
            self.model.eval()
            correct = total = 0

            # Loop through the validation data
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate validation accuracy
            val_acc = 100.0 * correct / total

            # Calculate elapsed time
            elapsed_time = int(time.time() - start_time)
            mins, secs = divmod(elapsed_time, 60)
            self.elapsed_str = f"{mins:02d}:{secs:02d}"
            
            # Emit a signal so the main thread can update the UI
            self.progress_signal.emit(epoch+1, avg_loss, val_acc)

        # Finished training — save the model if not stopped
        if not self._stop_flag:
            torch.save(self.model.state_dict(), self.model_file_path)
            self.finished_signal.emit(self.model_file_path)
        else:
            # If you want to handle partial training results, do so here
            self.finished_signal.emit("Training Stopped Early")



class TrainTab(QWidget):
    """
    A QWidget subclass for the training tab of the GUI.
    This tab allows the user to select a model, set training parameters,
    and start the training process.
    """

    def __init__(self, parent=None):
        """
        Initialize the training tab with a parent widget.
        Sets up the layout and UI elements.
        """

        super().__init__(parent)

        layout = QVBoxLayout()

        # Set the stylesheet for the entire widget
        self.title_box = QLabel("TRAIN")
        self.title_box.setObjectName("title_box")
        self.title_box.setAlignment(Qt.AlignCenter)
        self.title_box.setWordWrap(True)
        self.title_box.setFixedHeight(50)
        self.title_box.setFixedWidth(200)

        # Add the title label to the layout
        layout.addWidget(self.title_box, alignment=Qt.AlignCenter | Qt.AlignTop)
        
        # Add a label to select the model
        layout.addWidget(QLabel("Select Model"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["AlexNet", "InceptionV3", "Sesame 1.0"])
        layout.addWidget(self.model_dropdown)

        # Add a label to select the dataset
        ratio_layout = QVBoxLayout()
        ratio_label = QLabel("Train/Test Ratio")
        ratio_label.setAlignment(Qt.AlignVCenter)
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(50, 95)
        self.split_slider.setValue(80)
        self.split_value_label = QLabel(f"{self.split_slider.value()}%")
        self.split_value_label.setObjectName("split_value_label")
        self.split_slider.valueChanged.connect(lambda value: self.split_value_label.setText(f"{value}%"))

        # Add widgets to the layout
        ratio_layout.addWidget(ratio_label, alignment=Qt.AlignCenter)
        ratio_layout.addWidget(self.split_slider)
        ratio_layout.addWidget(self.split_value_label, alignment=Qt.AlignRight)
        layout.addLayout(ratio_layout)

        # Add a label to select the batch size
        batch_layout = QVBoxLayout()
        batch_label = QLabel("Batch size")
        batch_label.setAlignment(Qt.AlignCenter)
        self.batch_slider = QSlider(Qt.Horizontal)
        self.batch_slider.setRange(8, 128)
        self.batch_slider.setValue(32)
        self.batch_value_label = QLabel(f"{self.batch_slider.value()}")
        self.batch_value_label.setObjectName("split_value_label")
        self.batch_slider.valueChanged.connect(lambda value: self.batch_value_label.setText(str(value)))

        # Add widgets to the layout
        batch_layout.addWidget(batch_label, alignment=Qt.AlignCenter)
        batch_layout.addWidget(self.batch_slider)
        batch_layout.addWidget(self.batch_value_label, alignment=Qt.AlignRight)
        layout.addLayout(batch_layout)

        # Add a label to select the number of epochs
        epoch_layout = QVBoxLayout()
        epoch_label = QLabel("Epochs")
        epoch_label.setAlignment(Qt.AlignCenter)
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setRange(1, 100)
        self.epoch_slider.setValue(30)
        self.epoch_value_label = QLabel(f"{self.epoch_slider.value()}")
        self.epoch_value_label.setObjectName("split_value_label")
        self.epoch_slider.valueChanged.connect(lambda value: self.epoch_value_label.setText(str(value)))

        # Add widgets to the layout
        epoch_layout.addWidget(epoch_label, alignment=Qt.AlignCenter)
        epoch_layout.addWidget(self.epoch_slider)
        epoch_layout.addWidget(self.epoch_value_label, alignment=Qt.AlignRight)
        layout.addLayout(epoch_layout)

        # Add a button to start training
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button)

        # Add a button to load the dataset
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Add a label to show the training status
        self.status_label = QLabel("Ready to train.")
        layout.addWidget(self.status_label)

        # Add a canvas for plotting training progress
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add a button to stop training
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        layout.addWidget(self.stop_button)

        # Add a label to show the training status
        self.setLayout(layout)

        # Initialize instance variables
        self.trainer = None
        self.dataset_path = None
    
    def set_dataset_path(self, path):
        """
        Set the dataset path for training.
        This method is called when the user selects a dataset.
        """
        self.dataset_path = path
        print(f"[TrainTab] Dataset path set to: {path}")
        self.status_label.setText("Dataset loaded. Ready to train.")

    def start_training(self):
        """
        Start the training process based on user-selected parameters.
        This method is called when the user clicks the "Start Training" button.
        """
        if not self.dataset_path:
            self.status_label.setText("No dataset path. Please load data first.")
            return

        # Disable UI elements during training
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.model_dropdown.setEnabled(False)
        self.split_slider.setEnabled(False)
        self.batch_slider.setEnabled(False)
        self.epoch_slider.setEnabled(False)
        self.status_label.setText("Initializing training...")
        QApplication.processEvents()

        # Get user-selected parameters
        model_name = self.model_dropdown.currentText()
        split_ratio = self.split_slider.value()
        batch_size = self.batch_slider.value()
        epochs = self.epoch_slider.value()

        # Create device and store it as an instance attribute
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.device
        print(f"Training on {device} | Model: {model_name} | Split: {split_ratio}% | Batch: {batch_size} | Epochs: {epochs}")

        img_size = (224, 224) if model_name in ["AlexNet", "InceptionV3"] else (28, 28)

        # Define data transformations and augmentations
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ) if model_name in ["AlexNet", "InceptionV3"] else transforms.Normalize((0.5,), (0.5,))
        ])

        # Validation transform (no augmentations)
        val_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ) if model_name in ["AlexNet", "InceptionV3"] else transforms.Normalize((0.5,), (0.5,))
        ])

        # Load the dataset and split it into training and validation sets
        try:
            # Use ImageFolder to load the dataset
            t0 = time.time()
            full_dataset = ImageFolder(root=self.dataset_path)
            print("[TrainTab] class_to_idx:", full_dataset.class_to_idx)
            print(f"ImageFolder init took: {time.time() - t0:.2f}s")

            # Split the dataset into training and validation sets
            t1 = time.time()
            train_len = int(len(full_dataset) * (split_ratio / 100))
            val_len = len(full_dataset) - train_len
            train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            print(f"Split took: {time.time() - t1:.2f}s")
            
            self.val_dataset = val_dataset
            self.device = device


            # Apply transformations to the datasets
            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform

            # Create DataLoader for training and validation sets
            t2 = time.time()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=2, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True, persistent_workers=True)
            print(f"Dataloader init took: {time.time() - t2:.2f}s")
        
        except Exception as e:
            self.status_label.setText(f"Error loading dataset: {e}")
            self.train_button.setEnabled(True)
            return

        # Initialize the model
        model = get_model(model_name)
        model.to(device)

        # Set the model to training mode
        model_file_path = os.path.join("models", f"{model_name}_E{epochs}_B{batch_size}_S{split_ratio}.pt")
        os.makedirs("models", exist_ok=True)

        # Clear the previous plot and initialize new plot
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.train_losses = []
        self.val_accuracies = []

        # Initialize the progress bar and set its range
        self.progress_bar.setMaximum(epochs)
        self.progress_bar.setValue(0)

        # Start the training thread
        self.trainer = TrainerThread(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device,
            model_file_path=model_file_path
        )
        self.trainer.progress_signal.connect(self.handle_progress_update)
        self.trainer.finished_signal.connect(self.handle_training_finished)
        self.trainer.finished.connect(self.trainer.deleteLater)  # ✅ ensures cleanup

        self.trainer.start()

    def stop_training(self):
        """
        Stop the training process if it is currently running.
        This method is called when the user clicks the "Stop Training" button.
        """
        if self.trainer is not None:
            self.trainer.stop()
            self.status_label.setText("Stopping training... Please wait.")
            self.trainer.wait()
            self.trainer.deleteLater()
            self.status_label.setText("Training stopped.")
            self.trainer.finished_signal.connect(self.cleanup_after_stop)

    def cleanup_after_stop(self, msg):
        """
        Clean up after stopping the training process.
        This method is called when the training process is stopped.
        """
        if self.trainer:
            self.trainer.wait()
            self.trainer.deleteLater()
            self.trainer = None

        # Disable UI elements after stopping training
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.model_dropdown.setEnabled(True)
        self.split_slider.setEnabled(True)
        self.batch_slider.setEnabled(True)
        self.epoch_slider.setEnabled(True)
        self.status_label.setText("Training stopped.")


    def handle_progress_update(self, epoch, train_loss, val_acc):
        """
        Handle progress updates from the training thread.
        This method is called when the training thread emits a progress signal.
        """
        self.train_losses.append(train_loss)
        self.val_accuracies.append(val_acc)
        elapsed = getattr(self.trainer, 'elapsed_str', "--:--")

        # Update the progress bar and status label
        self.progress_bar.setValue(epoch)
        self.status_label.setText(
            f"Epoch {epoch}/{self.epoch_slider.value()} - Loss: {train_loss:.2f} - Val Acc: {val_acc:.2f}% - Elapsed: {elapsed}"
        )

        # Update the plot with the new data
        self.ax.clear()
        self.ax.plot(self.train_losses, label='Train Loss')
        self.ax.plot(self.val_accuracies, label='Val Accuracy')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Metric')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        self.canvas.draw()

    def handle_training_finished(self, msg):
        """
        Handle the completion of the training process.
        This method is called when the training thread emits a finished signal.
        """
        # Enable UI elements after training is finished
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.model_dropdown.setEnabled(True)
        self.split_slider.setEnabled(True)
        self.batch_slider.setEnabled(True)
        self.epoch_slider.setEnabled(True)

        # Update the status label with the final results
        if self.train_losses and self.val_accuracies:
            final_loss = self.train_losses[-1]
            final_acc = self.val_accuracies[-1]
            self.status_label.setText(f"Training finished. {msg}\nFinal Loss: {final_loss:.2f}, Final Val Acc: {final_acc:.2f}%")
        else:
            self.status_label.setText(f"Training finished. {msg}")
        
        # Clear the plot
        if self.trainer:
            self.trainer.wait()
            self.trainer.deleteLater()
            self.trainer = None

    def closeEvent(self, event):
        """
        Handle the close event for the training tab.
        This method is called when the user closes the tab.
        """
        if self.trainer and self.trainer.isRunning():
            print("[TrainTab] Waiting for training thread to finish...")
            self.trainer.stop()
            self.trainer.wait()
            self.trainer.deleteLater()
        event.accept()