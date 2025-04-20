import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import AlexNet_Weights


# Function to get an AlexNet model customized for 36 output classes
def get_alexnet(num_classes=36):
    model = models.alexnet(pretrained=False)  # Load AlexNet without pretrained weights
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # Modify final layer for classification
    return model

# Function to get an Inception-v3 model customized for 36 output classes
def get_inception_v3(num_classes=36):
    model = models.inception_v3(pretrained=False, aux_logits=False)  # Load InceptionV3, disable auxiliary outputs
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the final FC layer
    return model

class InceptionMiniBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionMiniBlock, self).__init__()

        # 1x1 Convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 3x3 Convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1)
        )

        # 5x5 Convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=1),
            nn.Conv2d(8, 16, kernel_size=5, padding=2)
        )

        # Pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 16, kernel_size=1)
        )

    def forward(self, x):
        # Apply each branch to the input tensor and concatenate results along the channel dimension
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

# SesameModel: A custom convolutional neural network designed for multi-class image classification.
# Combines architectural strengths from both AlexNet and Inception to balance speed, accuracy, and feature diversity.
#
# Key Features:
# - Starts with a wide 5x5 convolution + ReLU + MaxPool (inspired by AlexNet).
# - Includes a custom InceptionMiniBlock to capture multi-scale features via parallel convolution branches.
# - Uses Batch Normalization for faster convergence and stability.
# - Ends with a compact fully connected classifier using dropout for regularization.
# - Suitable for small to medium datasets where fast training and solid accuracy are needed.
class SesameModel(nn.Module):
    def __init__(self, num_classes=36):
        super(SesameModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # Like AlexNet's first layer
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 → 14x14

            InceptionMiniBlock(32),  # Output: 16+24+16+16 = 72 channels
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 → 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_sesame_1_0(num_classes=36):
    return SesameModel(num_classes)

# Function to dynamically get a model by name
def get_model(model_name, num_classes=36):
    if model_name == "AlexNet":
        return get_alexnet(num_classes)
    elif model_name == "InceptionV3":
        return get_inception_v3(num_classes)
    elif model_name == "Sesame 1.0":
        return get_sesame_1_0(num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
