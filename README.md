# SignCare - Sign Language Recognition System

SignCare is a sign language recognition system designed to facilitate communication between Deaf patients and healthcare providers. Using deep learning models, SignCare translates New Zealand Sign Language (NZSL) gestures into text in real-time, enhancing accessibility and communication.

## Features

- **Real-Time Sign Language Detection**: Recognizes sign language gestures through webcam input and provides immediate translation.
- **Multiple Model Support**: Choose from pre-trained models like AlexNet, InceptionV3, or the custom-built Sesame 1.0.
- **Customizable Training**: Set parameters like batch size, epochs, and train-test split to fine-tune model performance.
- **Prediction Probability Distribution**: Shows the confidence level of predictions and displays the top 5 predictions for each gesture.
- **Lightweight & Efficient**: Optimized to work efficiently on devices with limited computational resources.
- **User-Friendly Interface**: Built with PyQt5 for easy interaction with the system.

## Dependencies

The SignCare project requires the following libraries:

- **Python**: Programming language used for the project.
- **PyQt5**: For creating the graphical user interface (GUI).
- **TensorFlow**: For implementing deep learning models (CNNs) used in gesture recognition.
- **OpenCV**: For image and video processing, enabling webcam input and real-time sign recognition.
- **Matplotlib**: For visualizing training results and displaying charts.
- **Numpy**: For numerical operations, especially for handling datasets and training matrices.

## Version Details

- **Python version**: 3.9.21
- **PyQt5 version**: 5.15.10
- **TensorFlow version**: 2.5.0
- **OpenCV version**: 4.7.0.72
- **Matplotlib version**: 3.9.4
- **Numpy version**: 1.24.4

## Installation

To begin using the Sign Language Recognition System, make sure you have the correct prerequisites and dependencies installed on your machine.

Follow the steps below to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/project-deep-neural-network-python-dnn_team_20.git

2. **Navigate to the project directory**:
   ```bash
   cd project-deep-neural-network-python-dnn_team_20

3. **Compile and Run the application**:
   ```bash  
   python main.py  
