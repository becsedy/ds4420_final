# models/efficient_net.py
import timm
import torch.nn as nn

def create_efficientnet(model_name='efficientnet_b2', num_classes=10, pretrained=True):
    """
    Creates an EfficientNet model from the timm library.
    
    Args:
        model_name (str): The name of the EfficientNet variant (default 'efficientnet_b2').
        num_classes (int): Number of output classes (default 10).
        pretrained (bool): Use pretrained weights (default True).
        
    Returns:
        A torch.nn.Module instance of the EfficientNet model.
    """
    # Create the model; timm automatically creates the classifier head based on num_classes.
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

if __name__ == '__main__':
    # Quick test to verify model instantiation.
    model = create_efficientnet()
    print(model)