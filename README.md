# Face Detection Using YOLO Pretrained Model

## Overview

This project demonstrates how to train and evaluate face detection models using YOLO (You Only Look Once) with pretrained weights. Two different YOLO versions are compared: **YOLOv5** (with the WIDER FACE dataset) and **YOLOv11** (with the fareselmenshawii face detection dataset). The workflow includes data preparation, model training, evaluation, and visualization of results.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Comparison](#comparison)
- [References](#references)
- [Contact](#contact)

---

## Project Structure

```
Face-detection-using-Yolo-pre-trained-model/
│
├── Face_detection_yolov5.ipynb      # Notebook for YOLOv5 (WIDER FACE)
├── Face_dectection_yolov11.ipynb    # Notebook for YOLOv11 (fareselmenshawii dataset)
├── README.md                        # Project documentation
└── ...                              # Other files and folders
```

---

## Datasets

### 1. WIDER FACE (for YOLOv5)
- **Source:** [WIDER FACE on Kaggle](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark)
- **Description:** Large-scale, challenging dataset with diverse face images and annotations.

### 2. fareselmenshawii/face-detection-dataset (for YOLOv11)
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)
- **Description:** Smaller dataset for face detection tasks.

---

## Environment Setup

- Python 3.8+
- Jupyter Notebook
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Google Colab (recommended for GPU support)
- Required packages: `ultralytics`, `matplotlib`, `Pillow`, `kagglehub`, `torch`, etc.

Install Ultralytics YOLO:
```python
!pip install ultralytics
```

---

## Data Preparation

### For YOLOv5 (WIDER FACE)
- Download the dataset using `kagglehub`.
- Convert WIDER FACE annotations to YOLO format using the provided function.
- Organize images and labels into YOLO directory structure.
- Create a `data.yaml` file describing dataset paths and class names.

### For YOLOv11 (fareselmenshawii)
- Download the dataset using `kagglehub`.
- Create a `data.yaml` file for YOLO training.

---

## Training

### YOLOv5
- Load pretrained weights (`best_yolov5m.pt`).
- Train on the WIDER FACE dataset for 25 and 25 epochs.
- Use Adam optimizer, batch size 64, image size 320.

### YOLOv11
- Load pretrained weights (`best_yolov11m.pt`).
- Train on the fareselmenshawii dataset for 22 epochs.
- Use Adam optimizer, batch size 40, image size 320.

Both models save the best weights (`best.pt`) for evaluation and inference.

---

## Evaluation

- Evaluate the trained models on the validation set using:
  ```python
  metrics = model.val()
  ```
- Key metrics: **mAP (mean Average Precision), Precision, Recall, F1-score**.

---

## Visualization

- Display ground truth and predicted bounding boxes on validation images.
- Run inference on custom images and visualize detected faces with bounding boxes and confidence scores using `matplotlib`.

---

## Comparison

| Aspect         | YOLOv5 (WIDER FACE)                        | YOLOv11 (fareselmenshawii)           |
|----------------|--------------------------------------------|--------------------------------------|
| **Dataset**    | Large, diverse, challenging                | Smaller, less diverse                |
| **Epochs**     | 25 and 25                                  | 22                                   |
| **Batch Size** | 64                                         | 40                                   |
| **Optimizer**  | Adam                                       | Adam                                 |
| **Expected**   | Better generalization, robust performance  | Faster training, may overfit         |
| **Metrics**    | Use `model.val()` for mAP, Precision, etc. | Use `model.val()` for mAP, Precision |

**Note:** For a fair comparison, check the evaluation metrics and visualize predictions on the same test images.

---

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)
- [Kaggle: fareselmenshawii/face-detection-dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)
- [Kaggle: mksaad/wider-face-a-face-detection-benchmark](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark)

---

## Contact

Nguyen Tien Anh
- **Email:** anhnguyentien8365@gmail.com
