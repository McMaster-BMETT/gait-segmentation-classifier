# Gait Phase Segmentation Classifier

A deep learning pipeline for segmenting and classifying gait phases from wearable sensor data, built for knee rehabilitation monitoring. The system classifies 8-channel sensor streams (IMU, magnetometer, EMG) into 6 gait phases in real time using a Bidirectional Temporal Convolutional Network (BiTCN).

Developed as part of a McMaster BMETT capstone project.

## Overview

The classifier takes in time-series data from a wearable knee brace (ESP32-based, streaming via BLE at ~50 Hz) and segments each gait cycle into its constituent phases:

| Phase | Gait Cycle % |
|-------|-------------|
| Initial Contact | 0 -- 2% |
| Loading Response | 2 -- 12% |
| Mid Stance | 12 -- 31% |
| Terminal Stance | 31 -- 50% |
| Initial Swing | 50 -- 73% |
| Terminal Swing | 73 -- 100% |

Beyond classification, the pipeline computes clinically relevant rehabilitation metrics including Quadriceps Inhibition Index (QII), Hamstring-to-Quadriceps ratio (H:Q), phase-specific Range of Motion (ROM), and Limb Symmetry Index (LSI).

## Architecture

```
8-channel input (IMU roll/pitch/yaw, MAG x/y/z, EMG quad/ham)
        |
  Input Projection (1x1 Conv)
        |
  4x Residual TCN Blocks (dilations: 1, 2, 4, 8)
  ├── Dilated Conv1d (non-causal, symmetric padding)
  ├── BatchNorm + ReLU + Dropout
  ├── Dilated Conv1d
  ├── BatchNorm + ReLU + Dropout
  └── Residual Connection
        |
  Output Projection (1x1 Conv)
        |
  Per-timestep logits → 6 gait phase classes
```

- **Receptive field**: Covers the full gait cycle context via exponentially increasing dilations
- **Bidirectional**: Non-causal convolutions see both past and future context
- **Parameters**: ~100K (lightweight for edge deployment)

## Project Structure

```
gait-segmentation-classifier/
├── gait_classifier/
│   ├── models/
│   │   └── gait_tcn.py          # BiTCN model architecture
│   ├── data/
│   │   ├── synthetic.py         # Synthetic gait cycle generator
│   │   ├── healing_profiles.py  # 4-stage injury recovery profiles
│   │   ├── preprocessing.py     # Z-score normalization, EMG RMS envelope
│   │   └── dataset.py           # PyTorch Dataset + variable-length collation
│   ├── training/
│   │   ├── trainer.py           # Training loop (AdamW, cosine LR, early stopping)
│   │   ├── losses.py            # Weighted cross-entropy with label smoothing
│   │   └── metrics.py           # Per-phase precision, recall, F1
│   ├── clinical/
│   │   ├── quad_inhibition.py   # Quadriceps Inhibition Index
│   │   ├── hq_ratio.py          # Hamstring:Quadriceps ratio
│   │   ├── phase_rom.py         # Phase-specific range of motion
│   │   └── limb_symmetry.py     # Limb Symmetry Index
│   └── constants.py             # Gait phases, sensor specs, BLE UUIDs
├── demo/
│   ├── app.py                   # Dash web dashboard
│   ├── layout.py                # Dashboard layout components
│   ├── charts.py                # Plotly visualizations
│   └── callbacks.py             # Interactive callbacks
├── scripts/
│   ├── generate_data.py         # Synthetic dataset generation
│   ├── train.py                 # Model training entry point
│   ├── evaluate.py              # Test set evaluation
│   └── export_demo_data.py      # Export data for the demo dashboard
├── configs/
│   └── default.yaml             # Hyperparameters and sensor config
├── tests/
│   ├── test_model.py
│   ├── test_synthetic.py
│   ├── test_preprocessing.py
│   └── test_clinical_metrics.py
└── pyproject.toml
```

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gait-segmentation-classifier.git
cd gait-segmentation-classifier

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package with dependencies
pip install -e ".[dev]"
```

> **Note**: For CPU-only environments, install PyTorch without CUDA:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### Generate Synthetic Data

```bash
python scripts/generate_data.py --config configs/default.yaml
```

This creates training data across 4 healing stages (injured, early rehab, late rehab, healthy), with 200 gait cycles per stage by default.

### Train the Model

```bash
python scripts/train.py --config configs/default.yaml --save-path model.pth
```

Training uses:
- **Optimizer**: AdamW (lr=1e-3, weight decay=1e-4)
- **Scheduler**: Cosine annealing
- **Early stopping**: Patience of 15 epochs
- **Augmentation**: Time warping, Gaussian noise, EMG amplitude scaling
- **Loss**: Weighted cross-entropy with label smoothing (0.05)

### Evaluate

```bash
python scripts/evaluate.py
```

### Launch the Demo Dashboard

```bash
python scripts/export_demo_data.py   # generate demo data
python demo/app.py                    # open http://localhost:8050
```

The interactive Dash dashboard visualizes:
- Raw and classified sensor signals with phase-colored overlays
- Clinical metric progression across recovery stages
- Radar charts comparing patient metrics to healthy baselines

### Run Tests

```bash
pytest
```

33 tests covering model architecture, synthetic data generation, preprocessing, and clinical metric computation.

## Sensor Hardware

The wearable device is built on an **ESP32** microcontroller streaming 8 channels via BLE:

| Sensor | Channels | Details |
|--------|----------|---------|
| BNO08x IMU | Roll, Pitch, Yaw | Euler angles (0--360 degrees) |
| TLx493D Magnetometer | X, Y, Z | Magnetic flux density (mT) |
| EMG (ADC) | Quad, Hamstring | 12-bit ADC (0--4095 counts) |

- **Sampling rate**: ~50 Hz
- **BLE Service UUID**: `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`

## Clinical Metrics

| Metric | Description |
|--------|-------------|
| **Quadriceps Inhibition Index (QII)** | Ratio of quad activation during loading response vs. healthy baseline |
| **H:Q Ratio** | Hamstring-to-quadriceps co-contraction ratio (conventional and functional) |
| **Phase ROM** | Swing peak flexion and swing clearance angles |
| **Limb Symmetry Index (LSI)** | Bilateral and temporal symmetry across gait phases |

## Tech Stack

- **PyTorch** -- model architecture, training, and inference
- **ONNX / ONNX Runtime** -- model export for cross-platform deployment
- **NumPy / SciPy** -- signal processing and data generation
- **Dash / Plotly** -- interactive rehabilitation dashboard
- **Pytest** -- test suite

## Configuration

All hyperparameters are controlled via `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hidden_channels` | 64 | TCN hidden dimension |
| `model.num_blocks` | 4 | Number of residual TCN blocks |
| `model.dilations` | [1, 2, 4, 8] | Dilation factors per block |
| `training.epochs` | 100 | Maximum training epochs |
| `training.batch_size` | 64 | Training batch size |
| `training.learning_rate` | 1e-3 | Initial learning rate |
| `augmentation.time_warp_range` | 0.1 | Time warping +/-10% |
| `augmentation.gaussian_noise_std` | 0.02 | Gaussian noise std dev |

## License

McMaster BMETT 2025--26 Capstone Project.
