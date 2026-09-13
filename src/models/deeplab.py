from segmentation_models_pytorch import DeepLabV3

def build_deeplabv3(num_classes: int):
    return DeepLabV3(
        encoder_name='resnet34',
        num_classes=num_classes
    )