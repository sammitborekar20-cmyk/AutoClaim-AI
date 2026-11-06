# AutoClaim AI – Vehicle Damage Detection System For Insurance Claim

An AI-powered vehicle damage detection system that uses computer vision and deep learning to identify and localize different types of vehicle damage from images.

## Project Overview

AutoClaim AI is a computer vision-based solution designed to automate the initial identification of vehicle damage. It uses the YOLO11 object detection architecture to detect damage regions and classify them into 14 categories.

The system aims to support faster vehicle damage assessment by reducing the need for entirely manual visual inspection.

## Key Features

- **14-class damage detection:** Identifies multiple categories of vehicle damage.
- **Object localization:** Draws bounding boxes around detected damage regions.
- **Deep learning:** Uses the YOLO11 object detection architecture.
- **Dataset validation:** Includes scripts for checking annotation quality and dataset consistency.
- **Model evaluation:** Measures performance using precision, recall, and mean Average Precision (mAP).
- **Visual inspection:** Generates annotated images to inspect ground-truth labels and predictions.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Object Detection | Ultralytics YOLO11 |
| Computer Vision | OpenCV |
| Dataset | Roboflow Automobile Damage Detection |
| Data Analysis | NumPy, Pandas |
| Development Environment | VS Code, Python Virtual Environment |
| Hardware Acceleration | Apple Metal Performance Shaders (MPS) |

## Dataset

The project uses the Roboflow Automobile Damage Detection dataset in YOLO annotation format.

**Dataset statistics:**

| Dataset Split | Images | Labels |
|---|---:|---:|
| Training | 5,216 | 5,216 |
| Validation | 1,043 | 1,043 |
| Testing | 698 | 698 |
| **Total** | **6,957** | **6,957** |

The dataset contains 14 vehicle damage categories:

1. Bonnet dent
2. Boot dent
3. Door outer dent
4. Fender dent
5. Front bumper dent
6. Front windscreen damage
7. Headlight damage
8. Quarter panel dent
9. Rear bumper dent
10. Rear windscreen damage
11. Roof dent
12. Running board damage
13. Side mirror damage
14. Taillight damage

### Dataset Validation

Dataset inspection scripts were developed to verify:
- Image and annotation availability.
- Valid class IDs.
- Bounding-box coordinate values.
- Annotation consistency across training, validation, and test splits.
- Class distribution and potential class imbalance.

The annotation validation process reported zero invalid label lines, zero invalid class IDs, and zero invalid bounding-box values.

## Model Architecture

The project uses **YOLO11n (You Only Look Once)** for vehicle damage detection.

YOLO is an object detection architecture that predicts:
- The damage category.
- The confidence score.
- The bounding-box coordinates of the detected damage.

The model was trained using the Ultralytics framework with Apple MPS acceleration.

## Model Performance

The improved YOLO11n model was evaluated on the held-out test dataset.

| Evaluation Metric | Test Result |
|---|---:|
| Precision | **79.73%** |
| Recall | **64.89%** |
| mAP@0.50 | **74.73%** |
| mAP@0.50:0.95 | **45.53%** |

### Understanding the Metrics

- **Precision:** Measures how many predicted damage detections are correct.
- **Recall:** Measures how many actual damage instances are successfully detected.
- **mAP@0.50:** Measures average detection performance at an Intersection over Union threshold of 0.50.
- **mAP@0.50:0.95:** Measures detection performance across multiple IoU thresholds, providing a stricter evaluation.

## Project Structure

```text
AutoClaimAI/
│
├── app/                         # Application components
├── data/
│   └── raw/
│       └── data.yaml            # Dataset configuration
│
├── models/                      # Model-related files
│
├── reports/
│   └── boot_label_audit/        # Dataset inspection results
│
├── src/
│   ├── vision/
│   │   ├── train_detector.py
│   │   └── train_detector_improved.py
│   │
│   ├── audit_boot_labels.py
│   ├── check_dataset.py
│   ├── dataset_size.py
│   ├── dataset_storage.py
│   ├── evaluate_model.py
│   └── inspect_boot_damage.py
│
├── tests/                       # Testing scripts
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sammitborekar20-cmyk/AutoClaim-AI.git
cd AutoClaimAI
```



### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the Automobile Damage Detection dataset from Roboflow and arrange it according to the paths specified in:

```text
data/raw/data.yaml
```

The dataset is not included in this GitHub repository.

## Training and Evaluation

### Dataset validation

```bash
python src/check_dataset.py
```

### Dataset size analysis

```bash
python src/dataset_size.py
```

### Model training

```bash
python src/vision/train_detector.py
```

For improved-model fine-tuning:

```bash
python src/vision/train_detector_improved.py
```

### Model evaluation

```bash
python src/evaluate_model.py
```

Ensure the dataset paths and model checkpoint paths in the scripts are configured correctly before execution.

## Visual Results

The repository includes:
- `boot_damage_inspection.png` – boot damage dataset inspection.
- `reports/boot_label_audit/` – selected annotated boot damage samples.

These visualizations help inspect the quality of training annotations and understand the model's target damage regions.

## Challenges and Future Improvements

### Challenges
- Imbalanced distribution across damage categories.
- Lower detection performance for certain damage classes, particularly boot dents.
- Balancing precision and recall.
- Training deep learning models with limited local computing resources.

### Future Improvements
- Experiment with larger YOLO11 architectures.
- Improve detection of underrepresented damage categories.
- Explore class balancing and targeted data augmentation.
- Improve bounding-box localization.
- Integrate the trained model with a complete application interface.
- Explore damage severity estimation and automated claim assessment.

## Project Objective

The long-term goal of AutoClaim AI is to contribute to faster, more consistent, and intelligent vehicle damage assessment through computer vision and deep learning.

## Author

**Sammit Borekar**





---

