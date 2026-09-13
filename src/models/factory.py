from .deeplab import build_deeplabv3
from .segformer import build_segformer
from .unet import build_unet

def get_model(model_name: str, num_classes: int):
    if model_name.lower() == 'unet':
        return build_unet(num_classes=num_classes)

    elif model_name.lower() == 'segformer':
        return build_segformer(num_classse=num_classes)

    elif model_name.lower() == 'deeplabv3':
        return build_deeplabv3(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")