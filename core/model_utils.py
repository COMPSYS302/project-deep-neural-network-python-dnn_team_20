import torch
import torch.nn as nn
import torchvision.models as models

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

# Function to dynamically get a model by name
def get_model(model_name, num_classes=36):
    if model_name == "AlexNet":
        return get_alexnet(num_classes)
    elif model_name == "InceptionV3":
        return get_inception_v3(num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")  # Raise error if name is invalid
