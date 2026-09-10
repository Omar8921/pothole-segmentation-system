import cv2
import numpy as np
from pathlib import Path
import random, os
import matplotlib.pyplot as plt

from src.utils.image_io import read_grayscale_image, read_rgb_image


def validate_yolo_segmentation_label(label_path: Path):
    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    if not label_path.is_file():
        raise ValueError(f"Invalid label path: expected file, got directory: {label_path}")

    if label_path.suffix.lower() != ".txt":
        raise ValueError(
            f"Invalid label file extension: expected .txt, got {label_path.suffix}"
        )

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    report = {
        "empty_lines": [],
        "invalid_class_ids": [],
        "less_than_6_coordinate_lines": [],
        "odd_coordinate_lines": [],
        "out_of_range_coordinate_lines": [],
    }

    for line in lines:
        parts = line.split()

        if len(parts) == 0:
            report["empty_lines"].append(label_path)
            continue

        class_id_text, *coords_text = parts

        try:
            class_id = int(class_id_text)
        except ValueError:
            report["invalid_class_ids"].append(str(label_path))
            continue

        if class_id < 0:
            report["invalid_class_ids"].append(str(label_path))
            continue

        coords = []

        for coord_text in coords_text:
            try:
                coord = float(coord_text)
            except ValueError:
                report["out_of_range_coordinate_lines"].append(str(label_path))
                continue

            if coord < 0 or coord > 1:
                report["out_of_range_coordinate_lines"].append(str(label_path))

            coords.append(coord)

        if len(coords) < 6:
            report["less_than_6_coordinate_lines"].append(str(label_path))

        if len(coords) % 2 == 1:
            report["odd_coordinate_lines"].append(str(label_path))

    return report


def validate_mask(mask_path: Path, image_path: Path, allow_empty: bool = True):
    mask_path = Path(mask_path)
    image_path = Path(image_path)

    report = {
        "missing_masks": [],
        "unreadable_masks": [],
        "unreadable_images": [],
        "rgb_or_multichannel_masks": [],
        "size_mismatches": [],
        "not_binary_masks": [],
        "no_zero_background_masks": [],
    }

    if not mask_path.exists():
        report["missing_masks"].append(str(mask_path))
        return report

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        report["unreadable_masks"].append(str(mask_path))
        return report

    image = cv2.imread(str(image_path))

    if image is None:
        report["unreadable_images"].append(str(image_path))
        return report

    if mask.shape[:2] != image.shape[:2]:
        report["size_mismatches"].append({
            "image": str(image_path),
            "mask": str(mask_path),
            "image_shape": list(image.shape[:2]),
            "mask_shape": list(mask.shape[:2]),
        })

    if mask.ndim == 3:
        report["rgb_or_multichannel_masks"].append(str(mask_path))
        return report

    unique_values = np.unique(mask)

    if allow_empty and len(unique_values) == 1 and unique_values[0] == 0:
        return report

    if len(unique_values) != 2:
        report["not_binary_masks"].append({
            "mask": str(mask_path),
            "values": unique_values.tolist(),
        })

    elif 0 not in unique_values:
        report["no_zero_background_masks"].append({
            "mask": str(mask_path),
            "values": unique_values.tolist(),
        })

    return report

def masks_grid(image_paths: list[Path], row=5, col=5, shuffle=True):
    k = min(col * row, len(image_paths))

    if shuffle:
        sample = random.sample(image_paths, k=k)
    else:
        sample = image_paths[:k]        

    plt.figure(figsize=(4*col, 4*row))

    for i, image_path in enumerate(sample):
        image_name = image_path.name
        image = read_grayscale_image(image_path)

        plt.subplot(row, col, i+1)
        plt.title(image_name)
        plt.axis('off')
        plt.imshow(image)

    plt.show()


def generate_image_overlay(image_path: Path, mask_path: Path, alpha: float = 0.5) -> np.ndarray:
    image_path = Path(image_path)
    mask_path = Path(mask_path) 

    if not image_path.exists():
        raise FileNotFoundError(f'Image path does not exist: {image_path}')

    if not mask_path.exists():
        raise FileNotFoundError(f'Mask path does not exist: {mask_path}')

    image = read_rgb_image(image_path)
    mask = read_grayscale_image(mask_path)

    mask_bool = mask > 0

    overlay_color = np.zeros_like(image)
    overlay_color[mask_bool] = (0, 255, 0)

    overlay = image.copy()
    overlay[mask_bool] = (
        image[mask_bool] * (1 - alpha) + overlay_color[mask_bool] * alpha
    ).astype(np.uint8)

    return overlay


def masks_grid(image_paths: list[Path], row=5, col=5, shuffle=True):
    k = min(col * row, len(image_paths))

    if shuffle:
        sample = random.sample(image_paths, k=k)
    else:
        sample = image_paths[:k]        

    plt.figure(figsize=(4*col, 4*row))

    for i, image_path in enumerate(sample):
        image_name = image_path.name
        image = read_grayscale_image(image_path)

        plt.subplot(row, col, i+1)
        plt.title(image_name)
        plt.axis('off')
        plt.imshow(image)

    plt.show()


def generate_mask(image_path: Path, label_path: Path) -> np.ndarray:
    image_path = Path(image_path)
    label_path = Path(label_path)

    if not image_path.exists():
        raise FileNotFoundError(f'Image path does not exist: {image_path}')

    if not label_path.exists():
        raise FileNotFoundError(f'Label path does not exist: {label_path}')

    image = read_rgb_image(image_path)
    h, w, _ = image.shape

    mask = np.zeros((h, w), np.uint8)
    pothole_id = 1

    polygons = []

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        
        if len(parts) < 7:
            continue

        coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
        
        if len(coords) % 2 != 0:
            continue
        
        coords[::2] *= w
        coords[1::2] *= h

        polygon = np.round(coords).astype(np.int32).reshape(-1, 2)

        polygons.append(polygon)

    if polygons:
        cv2.fillPoly(mask, polygons, pothole_id)

    return mask


def generate_video_overlay_frames(
    video_path: Path,
    mask_path: Path,
    alpha: float = 0.25,
    verbose: bool = False
):
    if not 0 <= alpha <= 1:
        raise ValueError(f'Alpha must be between 0 and 1. Got {alpha}')

    video_path = Path(video_path)
    mask_path = Path(mask_path)

    if not video_path.exists():
        raise FileNotFoundError(f'Video path does not exist: {video_path}')

    video_ext = video_path.suffix

    if video_ext != '.mp4':
        raise ValueError(f'Invalid video extension. Expected .mp4 got {video_ext}')


    if not mask_path.exists():
        raise FileNotFoundError(f'Mask path does not exist: {mask_path}')

    mask_ext = mask_path.suffix

    if mask_ext != '.mp4':
        raise ValueError(f'Invalid video extension. Expected .mp4 got {mask_ext}')
    
    video = cv2.VideoCapture(str(video_path))
    mask = cv2.VideoCapture(str(mask_path))

    if not video.isOpened():
        raise ValueError(f'Could not open video: {video_path}')

    if not mask.isOpened():
        raise ValueError(f'Could not open mask video: {mask_path}')

    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    mask_width = int(mask.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask_height = int(mask.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask_fps = mask.get(cv2.CAP_PROP_FPS)
    mask_frame_count = int(mask.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print("Video:", video_width, video_height, video_fps, video_frame_count)
        print("Mask: ", mask_width, mask_height, mask_fps, mask_frame_count)

    if video_frame_count != mask_frame_count:
        raise ValueError(
            f'Video and mask have different frame counts: '
            f'{video_frame_count} vs {mask_frame_count}'
        )
    
    if (video_width, video_height) != (mask_width, mask_height):
        raise ValueError(
            f'Video and mask sizes do not match: '
            f'{video_width}x{video_height} vs {mask_width}x{mask_height}'
        )

    while True:
        video_success, video_frame = video.read()
        mask_success, mask_frame = mask.read()

        if not video_success or not mask_success:
            break
    
        if mask_frame.shape[:2] != video_frame.shape[:2]:
            height, width = video_frame.shape[:2]
            mask_frame = cv2.resize(
                mask_frame,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            ) 
        
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

        mask_bool = mask_gray > 0

        overlay_color = np.zeros_like(video_frame)
        overlay_color[mask_bool] = (0, 255, 0)

        overlay_frame = video_frame.copy()
        overlay_frame[mask_bool] = (
            video_frame[mask_bool] * (1 - alpha) +
            overlay_color[mask_bool] * alpha
        ).astype(np.uint8)

        yield overlay_frame
    
    video.release()
    mask.release()


def mask_to_yolo(    
    mask_path: Path,
    output_dir: Path,
    name: str,
    class_id: int = 0,
    epsilon_ratio: float = 0.002) -> Path:

    mask_path = Path(mask_path)

    if not mask_path.is_file():
        raise IsADirectoryError(f'Expected image file, but got a directory: {mask_path}')

    if not mask_path.exists():
        raise FileNotFoundError(f'Image file could not be found: {mask_path}')
    
    mask = read_grayscale_image(mask_path)
    h, w = mask.shape

    if mask is None:
        raise ValueError(f"File exists but could not be read as an image: {mask_path}")

    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    yolo_lines = []

    for contour in contours:

        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        if len(polygon) < 3:
            continue

        normalized_points = []

        for point in polygon:
            x, y = point[0]

            x_normalized = x / w
            y_normalized = y / h

            normalized_points.extend([
                f"{x_normalized:.6f}",
                f"{y_normalized:.6f}"
            ])

        line = f"{class_id} " + " ".join(normalized_points)
        yolo_lines.append(line)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / name, "w", encoding="utf-8") as file:
        file.write("\n".join(yolo_lines))
