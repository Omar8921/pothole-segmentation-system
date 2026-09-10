# Data Inspection and Preprocessing

This document describes the inspection, cleaning, annotation correction, and preprocessing pipeline used to prepare the pothole semantic segmentation dataset for model training and evaluation.

## Overview

The final dataset was created by combining three public pothole datasets with different annotation formats and levels of annotation quality. Before training, the data was inspected, cleaned, standardized, and split into training, validation, and testing subsets.

The pipeline can be summarized as:

```text
Raw datasets
    ↓
Format inspection
    ↓
Annotation validation
    ↓
Manual image/mask review
    ↓
Mask correction in CVAT
    ↓
Remove unusable samples
    ↓
Standardize image/mask format
    ↓
Resize to 512 × 512
    ↓
Create train/validation/test splits
    ↓
Model-ready dataset
```

---

## 1. Dataset Inspection

The source datasets were first inspected to understand:

- Image formats and resolutions
- Annotation formats
- Image-mask correspondence
- Missing or invalid labels
- Incorrect or poor-quality segmentation masks
- Duplicate or unusable samples
- Differences in mask encoding between datasets

Because the datasets came from different sources, their annotations could not be used directly without validation and standardization.

---

## 2. Annotation Validation

Automated validation was performed before manual inspection.

For YOLO segmentation annotations, the validation process checked for issues such as:

- Missing label files
- Invalid file extensions
- Empty annotation lines
- Invalid class IDs
- Too few polygon coordinates
- Odd numbers of polygon coordinates
- Coordinates outside the normalized `[0, 1]` range

Image-mask pairs were also checked to make sure that the required files existed and corresponded correctly.

---

## 3. Annotation Review Tool

A custom annotation inspection tool was used to manually review the dataset.

Each image and its corresponding segmentation mask were inspected together and assigned one of three labels:

- **OK** — annotation is acceptable and can be kept.
- **Must Fix** — annotation contains an error but can be manually corrected.
- **Must Delete** — sample is unsuitable for the dataset and should be removed.

This inspection stage was used to avoid blindly trusting annotations from the original public datasets.

---

## 4. Manual Mask Correction with CVAT

Samples marked **Must Fix** were corrected manually using **CVAT**.

CVAT was run locally using Docker and used to redraw or repair inaccurate pothole segmentation masks.

After correction, the fixed masks were stored in the interim data directory and later incorporated into the cleaned dataset.

Samples marked **Must Delete** were excluded from the final dataset.

A total of **1,392 unusable samples were removed** during the cleaning process.

---

## 5. Annotation Format Standardization

The source datasets used different annotation formats, so all annotations were converted into a common binary semantic-segmentation mask representation.

### YOLO Segmentation Labels

Datasets containing YOLO polygon annotations were converted from normalized polygon coordinates into pixel-level binary masks.

### Existing Segmentation Masks

Existing masks were standardized so that each sample contained:

- Background pixels
- Pothole pixels

Masks were converted to a consistent single-channel binary representation before training.

---

## 6. Video Dataset Processing

One of the source datasets provided road footage together with corresponding segmentation-mask videos.

The videos were processed by extracting synchronized frames from:

- The original road video
- The corresponding mask video

Frame alignment was preserved so that every extracted RGB frame matched its correct segmentation mask.

To reduce data leakage, frames originating from the same video sequence were kept together when creating dataset splits rather than being randomly distributed across training and testing subsets.

---

## 7. Train, Validation, and Test Split

After cleaning and preprocessing, the final dataset contained **4,054 image-mask pairs**.

| Split | Samples |
|---|---:|
| Training | 2,749 |
| Validation | 654 |
| Testing | 651 |
| **Total** | **4,054** |

The split assignments are stored in:

```text
splits.csv
```

The file provides the mapping between each image, its corresponding mask, and its assigned dataset split.

---

## 9. Final Dataset Structure

The cleaned dataset follows a simple image-mask structure:

```text
processed/
├── images/
├── masks/
└── splits.csv
```

### `images/`

Contains the cleaned and standardized RGB road images.

### `masks/`

Contains the corresponding binary pothole segmentation masks.

### `splits.csv`

Stores the train, validation, and test assignment for every image-mask pair.

---

## Result

The preprocessing pipeline produces a cleaned, standardized, and manually reviewed semantic-segmentation dataset suitable for training deep learning models such as U-Net and DeepLabV3+.

The main goals of the pipeline were to:

- Improve annotation quality
- Remove unusable samples
- Standardize heterogeneous datasets
- Prevent train/test leakage
- Preserve segmentation-mask integrity
- Produce a reproducible dataset structure for model development
