from segmentation_models_pytorch import Segformer

def build_segformer(num_classse: int):
    return Segformer(
        encoder_name='resnet34',
        num_classse=num_classse
    )