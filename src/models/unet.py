from segmentation_models_pytorch import Unet

def build_unet(num_classes: int):
    return Unet(
        encoder_name='resnet34',
        num_classes=num_classes
    )