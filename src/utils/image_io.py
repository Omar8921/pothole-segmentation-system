from pathlib import Path
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from typing import Iterable

def save_video_frames(
    frames: Iterable[np.ndarray],
    output_dir: Path,
    name: str,
    fps: float,
    frame_size: tuple[int, int] | None = None
) -> Path:
    '''
    Save video frames to disk and return the saved video path.
    '''

    output_dir = Path(output_dir)

    # Ensure output_dir is actually a directory
    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            f'Expected a directory path, but got a file path: {output_dir}'
        )

    # Validate frames
    if frames is None:
        raise ValueError('Frames is None')

    # Validate FPS
    if fps is None or fps <= 0:
        raise ValueError(f'FPS must be greater than 0. Got {fps}')

    # Add .mp4 extension if missing
    if Path(name).suffix == '':
        name += '.mp4'

    name = Path(name).name

    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / name

    frames_iter = iter(frames)

    try:
        first_frame = next(frames_iter)
    except StopIteration:
        raise ValueError('No frames were provided')

    if frame_size is None:
        height, width = first_frame.shape[:2]
        frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size
    )

    if not writer.isOpened():
        raise IOError(f'Failed to create video writer for: {output_path}')

    writer.write(first_frame)

    for frame in frames_iter:
        writer.write(frame)

    writer.release()

    return output_path

def save_rgb_image(image: np.ndarray, output_dir: Path, name: str) -> Path:
    """
    Save an RGB image to disk and return the saved image path.

    Expected shape:
        (H, W, 3)

    Note:
        OpenCV saves color images in BGR format, so RGB is converted to BGR before saving.
    """

    output_dir = Path(output_dir)

    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            f"Expected a directory path, but got a file path: {output_dir}"
        )

    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected image to be np.ndarray, got {type(image)}"
        )

    if image.size == 0:
        raise ValueError("Image array is empty")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {image.shape}"
        )

    if Path(name).suffix == "":
        name += ".png"

    name = Path(name).name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / name

    # Convert RGB to BGR because cv2.imwrite expects BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), image_bgr)

    if not success:
        raise IOError(f"Failed to save RGB image to: {output_path}")

    return output_path


def save_grayscale_image(image: np.ndarray, output_dir: Path, name: str) -> Path:
    """
    Save a grayscale image to disk and return the saved image path.

    Expected shape:
        (H, W)
    """

    output_dir = Path(output_dir)

    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            f"Expected a directory path, but got a file path: {output_dir}"
        )

    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected image to be np.ndarray, got {type(image)}"
        )

    if image.size == 0:
        raise ValueError("Image array is empty")

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image with shape (H, W), got {image.shape}"
        )

    if Path(name).suffix == "":
        name += ".png"

    name = Path(name).name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / name

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise IOError(f"Failed to save grayscale image to: {output_path}")

    return output_path


def save_image(image: np.ndarray, output_dir: Path, name: str, color_order: str = "RGB") -> Path:
    '''
    Save an image to disk and return the saved image path.
    '''

    output_dir = Path(output_dir)

    # Ensure output_dir is actually a directory
    if output_dir.exists() and output_dir.is_file():
        raise ValueError(
            f'Expected a directory path, but got a file path: {output_dir}'
        )

    # Validate image
    if image is None:
        raise ValueError('Image is None')

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f'Expected image to be np.ndarray, got {type(image)}'
        )

    if image.size == 0:
        raise ValueError('Image array is empty')

    if Path(name).suffix == '':
        name += '.png'

    name = Path(name).name

    os.makedirs(output_dir, exist_ok=True)

    image_to_save = image

    # OpenCV writes 3-channel images as BGR, so convert RGB before saving.
    if image.ndim == 3 and image.shape[2] == 3:
        if color_order.upper() == "RGB":
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_order.upper() == "BGR":
            image_to_save = image
        else:
            raise ValueError("color_order must be either 'RGB' or 'BGR'")


    output_path = output_dir / name

    success = cv2.imwrite(str(output_path), image_to_save)

    if not success:
        raise IOError(f'Failed to save image to: {output_path}')

    return output_path


def read_rgb_image(image_path: Path) -> np.ndarray:
    """
    Read an RGB image from a path and return it as a NumPy array.
    """

    # Convert input to Path so the function accepts both strings and Path objects.
    image_path = Path(image_path)

    # Check if the path exists before trying to read it.
    # This gives a clearer error than just saying cv2.imread returned None.
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    # Check that the path is actually a file, not a folder.
    if not image_path.is_file():
        raise IsADirectoryError(f"Expected an image file, but got a directory: {image_path}")

    # Convert Path to string because cv2.imread is safest with string paths.
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # If OpenCV still cannot read it, the file may be corrupted or unsupported.
    if img is None:
        raise ValueError(f"File exists but could not be read as an image: {image_path}")

    # Convert from OpenCV's default BGR format to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def convert_to_binary(image_path: Path) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.is_file():
        raise IsADirectoryError(f'Expected image file, but got a directory: {image_path}')

    if not image_path.exists():
        raise FileNotFoundError(f'File could not be found: {image_path}')
    
    mask = read_rgb_image(image_path)

    if mask is None:
        raise ValueError(f'Image exists but could not be read: {image_path}')
    
    if len(mask.shape) == 2:
        binary_mask = (mask > 0).astype(np.uint8)
    
    elif len(mask.shape) == 3:
        binary_mask = np.any(mask > 0, axis=2).astype(np.uint8)

    return binary_mask


def read_grayscale_image(image_path: Path) -> np.ndarray:
    """
    Read a grayscale image from a path and return it as a NumPy array.
    """

    # Convert input to Path so the function accepts both strings and Path objects.
    image_path = Path(image_path)

    # Check if the path exists.
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    # Check that the path is a file.
    if not image_path.is_file():
        raise IsADirectoryError(f"Expected an image file, but got a directory: {image_path}")

    # Convert Path to string for OpenCV compatibility.
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # If OpenCV cannot read the file, it is probably corrupted or unsupported.
    if img is None:
        raise ValueError(f"File exists but could not be read as a grayscale image: {image_path}")

    return img


def images_grid(image_paths: list[Path], row=5, col=5, shuffle=True):
    k = min(col * row, len(image_paths))

    if shuffle:
        sample = random.sample(image_paths, k=k)
    else:
        sample = image_paths[:k]        

    plt.figure(figsize=(4*col, 4*row))

    for i, image_path in enumerate(sample):
        image_name = image_path.name
        image = read_rgb_image(image_path)

        plt.subplot(row, col, i+1)
        plt.title(image_name)
        plt.axis('off')
        plt.imshow(image)

    plt.show() 