import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import AlexNet_Weights


# Function to get an AlexNet model customized for 36 output classes
def get_alexnet(num_classes=36):
    # 1. Load pretrained AlexNet with ImageNet weights
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    # 2. Add BatchNorm after each Conv2d layer in features
    new_features = nn.Sequential()
    for i, layer in enumerate(model.features):
        new_features.add_module(str(i), layer)
        if isinstance(layer, nn.Conv2d):
            new_features.add_module(f'bn{i}', nn.BatchNorm2d(layer.out_channels))
    model.features = new_features

    # 3. Remove Dropout in classifier (0 and 3), keep ReLU and Linear
    model.classifier[0] = nn.Identity()  # replaces Dropout
    model.classifier[3] = nn.Identity()  # replaces Dropout

    # 4. Replace final classification layer
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # 5. Use channels-last format for faster convolutions on GPU
    return model.to(memory_format=torch.channels_last)

# Function to get an Inception-v3 model customized for 36 output classes
def get_inception_v3(num_classes=36):
    model = models.inception_v3(pretrained=False, aux_logits=False)  # Load InceptionV3, disable auxiliary outputs
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the final FC layer
    return model

# Define a super light custom model called Sesame 1.0
class SesameModel(nn.Module):
    def __init__(self, num_classes=36):
        super(SesameModel, self).__init__()
        # First convolution: input channels=3, output channels=16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolution: input channels=16, output channels=32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Assuming input images are 28x28, after two poolings the feature map becomes 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
