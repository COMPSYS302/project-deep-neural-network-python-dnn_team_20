# SignCare - Sign Language Recognition System

SignCare is a sign language recognition system designed to facilitate communication between Deaf patients and healthcare providers. Using deep learning models, SignCare translates ASL gestures into text in real-time, enhancing accessibility and communication.

## 🚀 Features

- **Real-Time Sign Language Detection** via webcam
- **Multiple Model Support:** AlexNet, InceptionV3, and custom-built Sesame 1.0
- **Custom Training Parameters:** Batch size, epochs, train/test split
- **Prediction Confidence:** View top 5 predictions and their probabilities
- **Optimized for Efficiency** on low-resource devices
- **User-Friendly Interface** built with PyQt5

## 📦 Project Environment Setup Instructions
SignCare requires **Anaconda** and **Pip** to manage the installation of key libraries including PyTorch, PyQt5, NumPy, and OpenCV.

Anaconda allows you to create isolated environments so that package installations do not interfere with other projects. Pip installs packages within the currently active Conda environment.

> 💡 Use `conda list` to view all packages installed in your current environment.

1. **Install anaconda**:
   a. Download for your operating system https://www.anaconda.com/download
   b. Follow installation instructions. If you are on linux, you need to make the anaconda.sh file executable, then run “./anaconda.sh” in the terminal.
   ```bash
   conda create --name cs302 python=3.9
   ```
   c.	In the Anaconda prompt (windows) or a new terminal (linux/osx), create a new conda environment conda create --name cs302 python=3.9
   ```bash
   conda activate cs302
   ```
   d.	Activate your environment conda activate cs302. You will need to repeat this step every time you want to activate your environment. 

2. **Install Pytorch**:
   a.	Navigate to https://pytorch.org/
   b.	Under the Install PyTorch section, select the stable build,  the OS you are using, Conda for type of package and Python for type of language. If you have a Nvidia GPU, select CUDA 11.8 for the compute platform, otherwise select CPU.
   c.	Copy and paste the command into your anaconda prompt/terminal. Ensure you are still in the cs302 environment. This will take some time.
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 0c pytorch -c nvidia
   ```
   
3.** Install Remaining Packages**
      In your cs302 environment, use pip to install the correct versions of opencv, pyqt and numpy:

   ```bash
   pip install open-cv==4.7.0.72
   ```
   ```bash
   pip install PyQt5==5.15.10
   ```
   ```bash
   pip install numpy==1.24.4
   ```
4. ** Verify installation **
   a.	In your conda environment, enter the following commands:
   ```bash
   python
   import torch
   ```
   b.	There should be no errors. If you installed pytorch with CUDA, ```bash torch.cuda.is_available() ```
   should return true
   c.	Type    ```bash exit() ```to exit the python interpreter

5. Install IDE (Visual studio code)
   a.	Visit https://code.visualstudio.com/ to download and follow the installation instructions.

## 📚 Libraries

- **Python version**: 3.9.21
- **PyQt5 version**: 5.15.10
- **TensorFlow version**: 2.5.0
- **OpenCV version**: 4.7.0.72
- **Matplotlib version**: 3.9.4
- **Numpy version**: 1.24.4

## 🔧 Installation

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
