import os
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QScrollArea, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ViewDataTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Layout to hold all the widgets
        layout = QVBoxLayout()

        self.title_box = QLabel("VIEW")
        self.title_box.setObjectName("title_box")
        self.title_box.setAlignment(Qt.AlignCenter)  # Center the text inside the box
        self.title_box.setWordWrap(True)  # Allow text to wrap

        self.title_box.setFixedHeight(50)  # Set height of the title box
        self.title_box.setFixedWidth(200)  # Set width of the title box

        # Add the title label to the layout
        layout.addWidget(self.title_box, alignment=Qt.AlignCenter | Qt.AlignTop) 


        # Input field for the user to type a label to filter images (A-Z or 0-9)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter sign label to filter (A-Z or 0-9)")
        self.filter_input.textChanged.connect(self.update_filter)  # Connect filter input change to the update function
        layout.addWidget(self.filter_input)

        # Scrollable area to display images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Allow the widget to resize
        self.scroll_area.setMinimumSize(600, 250)  # Set minimum size for the scroll area
        self.scroll_widget = QWidget()  # Widget to hold all the images
        self.scroll_layout = QGridLayout()  # Grid layout to arrange the images
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        layout.addWidget(self.scroll_area)  # Add the scrollable area to the layout

                # Label for dataset statistics
        self.stats_label = QLabel("Dataset Statistics:\n")
        layout.addWidget(self.stats_label)

        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(100) 
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        # Set the layout for this widget
        self.setLayout(layout)
        
        
        # Initialize variables for storing image folder path and image data
        self.image_folder = None
        self.image_data = []

    def load_images(self, folder_path):
        """Load images from the provided folder path."""
        self.image_folder = folder_path
        self.image_data.clear()  # Clear the current image data

        # Remove all existing widgets from the scroll layout
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)

        # Check if the folder path exists
        if not os.path.isdir(folder_path):
            return

        # Iterate through all class folders (36 labels)
        for class_label in range(36):
            class_folder = os.path.join(folder_path, f"{class_label:02d}")
            if os.path.isdir(class_folder):
                # For each image in the class folder, add to the image data list
                for img_file in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_file)
                    self.image_data.append((img_path, str(class_label)))

        # Update dataset statistics and display the images
        self.update_statistics()
        self.display_images()

    def display_images(self, filter_label=None):
        """Display images in the scroll area, filtered by the provided label."""
        # Remove all existing widgets from the scroll layout
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)
        
        if filter_label == None:
            return
            
        # Filter images based on the provided label (if any)
        filtered_data = [img for img in self.image_data if filter_label is None or img[1] == filter_label]


        # Display the filtered images in the grid layout
        for i, (img_path, _) in enumerate(filtered_data):  
            pixmap = self.load_pixmap(img_path)
            if pixmap:
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                self.scroll_layout.addWidget(img_label, i // 5, i % 5)  # Place the image in the grid

    def update_filter(self):
        filter_text = self.filter_input.text().strip().lower()

        if filter_text.isdigit():
            num = int(filter_text)
            if 0 <= num <= 9:
                label = str(num + 26)  # Map numbers 0-9 to classes 26-35
                self.display_images(filter_label=label)
            else:
                self.display_images(filter_label=filter_text)  # Just use the digit directly
        elif filter_text.isalpha() and len(filter_text) == 1:
            label = str(ord(filter_text) - ord('a'))  # Convert 'a' -> 0, ..., 'z' -> 25
            self.display_images(filter_label=label)
        else:
            self.display_images(filter_label=None)  # Show all images if input is invalid


    def update_statistics(self):
        """Update the statistics label with the count of images for each class."""
        label_counts = {label: 0 for label in range(36)}  # Initialize label counts
        for _, label in self.image_data:
            label_counts[int(label)] += 1  # Increment the count for the appropriate label
        # Format the statistics text
        # stats_text = "Dataset Statistics:\n" + "\n".join(f"{k}: {v} images" for k, v in label_counts.items())
        # self.stats_label.setText(stats_text)  # Update the stats label
        self.plot_statistics(label_counts)

    def plot_statistics(self, label_counts):
        """Plot a bar chart of image counts using Matplotlib."""
        self.figure.clear()  # Clear previous figure

        ax = self.figure.add_subplot(111)
        ax.bar(label_counts.keys(), label_counts.values(), color='skyblue')

        ax.set_xlabel("Class Label")
        ax.set_ylabel("Number of Images")
        ax.set_title("Image Count per Class")
        ax.set_xticks(range(0, 36, max(1, 36 // 10)))  # Reduce number of x-axis labels if too many

        self.canvas.draw()


    def load_pixmap(self, img_path):
        # Load the image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Resize the image (e.g., to 100x100 pixels) for compression
        img = cv2.resize(img, (100, 100))  # You can adjust the size here
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Qt
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        return QPixmap.fromImage(q_img).scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio)

