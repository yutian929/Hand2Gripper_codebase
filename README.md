# Hand2Gripper — Codebase (ECCV 2026 Supplementary Material)

Official implementation of **"Hand2Gripper: A Low-Cost Data Generation Framework for Imitation Learning with the Adaptive Hand-Gripper Mapping Model"**.

> **Note:** This codebase is provided as supplementary material for double-blind review.  
> Some submodules are **not included** due to file-size limits and the double-blind policy.  
> Please refer to [`submodules/README.md`](submodules/README.md) for download and setup instructions of the missing submodules.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Submodules](#submodules)

---

## Overview

Hand2Gripper is a framework that converts human hand demonstrations into gripper-executable actions for robot imitation learning. The system consists of:

- **Data Processing Pipeline** — Processes raw egocentric video data through a series of stages (hand detection, segmentation, 3D pose estimation, action extraction, inpainting, etc.)
- **Hand-Gripper Mapping Model** — An adaptive model that maps hand poses/actions to gripper commands, supporting multiple gripper types.
- **Digital Twin System** — MuJoCo-based simulation for rendering robot overlays and verifying mapped actions.
- **Visual SLAM** — Camera trajectory estimation for egocentric video sequences.
- **Annotation Tool** — Interactive tool for reviewing and correcting the hand-gripper mapping results.

---

## Repository Structure

```
Hand2Gripper_codebase/
├── README.md                  # This file
├── install.sh                 # Full installation script
├── setup.py                   # Package setup
├── source.sh                  # Environment sourcing script
├── configs/
│   ├── default.yaml           # Default configuration
│   ├── epic.yaml              # EPIC-Kitchens / custom dataset configuration
│   └── sam2_hiera_l.yaml      # SAM2 model configuration
├── hand2gripper/              # Main package
│   ├── __init__.py
│   ├── hand.py                # Hand model (unconstrained & physically constrained)
│   ├── process_data.py        # Main data processing entry point
│   ├── hand2gripper_annotator.py  # Annotation tool entry point
│   ├── twin_robot.py          # Single-arm digital twin (MuJoCo)
│   ├── twin_bimanual_robot.py # Bimanual digital twin (MuJoCo)
│   ├── camera/                # Camera intrinsics/extrinsics JSON configs
│   ├── detectors/             # Detection backends
│   │   ├── detector_hamer.py  # HaMeR hand mesh recovery
│   │   ├── detector_dino.py   # Grounding DINO object detection
│   │   ├── detector_sam2.py   # SAM2 segmentation
│   │   └── detector_detectron2.py  # Detectron2 instance segmentation
│   ├── processors/            # Processing pipeline stages
│   │   ├── base_processor.py  # Base class with shared logic
│   │   ├── bbox_processor.py  # Hand bounding-box detection
│   │   ├── hand_processor.py  # 2D/3D hand pose estimation
│   │   ├── segmentation_processor.py  # Hand & arm segmentation
│   │   ├── action_processor.py        # Action extraction & VSLAM
│   │   ├── smoothing_processor.py     # Trajectory smoothing
│   │   ├── handinpaint_processor.py   # Hand region inpainting (E2FGVI)
│   │   ├── robotinpaint_processor.py  # Robot overlay rendering
│   │   ├── hand2gripper_annotator_processor.py  # Annotation processor
│   │   ├── paths.py           # Path management
│   │   └── phantom_data.py    # Data structures
│   └── utils/                 # Utility functions
│       ├── bbox_utils.py
│       ├── data_utils.py
│       ├── image_utils.py
│       ├── pcd_utils.py
│       ├── transform_utils.py
│       └── hand2gripper_types_hand_detection.py
└── submodules/                # External dependencies (see submodules/README.md)
    ├── README.md              # Submodule download & setup instructions
    ├── arx_mujoco/            # ARX R5 robot MuJoCo interface (included)
    ├── Hand2Gripper_model/    # Hand-Gripper mapping model (included)
    ├── Hand2Gripper_vslam/    # Visual SLAM module (included)
    └── (other submodules)     # See submodules/README.md for setup
```

---

## Prerequisites

- **OS:** Ubuntu 20.04+ (tested on Ubuntu 22.04)
- **Python:** 3.10
- **CUDA:** 12.1
- **GPU:** NVIDIA GPU with ≥ 8 GB VRAM (tested on RTX 3090 / A100)
- **Conda:** Miniconda or Anaconda

---

## Installation

### 1. Set Up Missing Submodules

Before running `install.sh`, you must download and place the missing submodules.  
See [`submodules/README.md`](submodules/README.md) for detailed instructions.

### 2. Run the Installation Script

```bash
# This creates the conda environment and installs all dependencies
bash install.sh
```

### 3. Source the Environment

```bash
conda activate phantom
source source.sh
```

---

## Configuration

Configuration is managed via [Hydra](https://hydra.cc/) YAML files in `configs/`:

| Config | Description |
|---|---|
| `default.yaml` | Default settings (single-arm, Panda robot, Robotiq85 gripper) |
| `epic.yaml` | EPIC-Kitchens and custom egocentric dataset settings (bimanual ARX R5, HACO) |
| `sam2_hiera_l.yaml` | SAM2 Hiera-Large model architecture config |

Key configurable parameters:

```yaml
mode: ["all"]            # Processing stages: bbox, hand2d, hand3d, hand_segmentation,
                         # arm_segmentation, action, smoothing, hand_inpaint, robot_inpaint, all
data_root_dir: "..."     # Input raw data directory
processed_data_root_dir: "..."  # Output processed data directory
robot: "Arx5"            # Robot type: Panda, Arx5
gripper: "Robotiq85"     # Gripper type
hand_model: "hamer"      # Hand model: hamer, wilor
bimanual_setup: "shoulders"  # Setup: single_arm, shoulders
target_hand: "both"      # Which hand(s) to process: left, right, both
```

---

## Usage

### Data Processing Pipeline

Process raw demonstration videos through the full pipeline:

```bash
# Process all demos with the EPIC/custom config
python -m hand2gripper.process_data --config-name epic

# Process a specific demo
python -m hand2gripper.process_data --config-name epic demo_name=demo demo_num=0

# Run only specific processing stages
python -m hand2gripper.process_data --config-name epic mode='["bbox","hand2d"]'
```

The processing pipeline runs in the following order:
1. **bbox** — Hand bounding-box detection (DINO / EPIC-Kitchens annotations)
2. **hand2d** — 2D hand keypoint detection (HaMeR / WiLoR)
3. **hand_segmentation** — Hand region segmentation (SAM2 + Detectron2)
4. **arm_segmentation** — Arm region segmentation
5. **hand3d** — 3D hand pose estimation with depth
6. **action** — Action extraction (hand-gripper mapping + VSLAM camera trajectory)
7. **smoothing** — Trajectory smoothing
8. **hand_inpaint** — Remove human hands from video (E2FGVI)
9. **robot_inpaint** — Render robot overlay in the inpainted video (MuJoCo digital twin)

### Hand-Gripper Annotation Tool

Review and correct the mapping results interactively:

```bash
python -m hand2gripper.hand2gripper_annotator --config-name epic
```

---

## Submodules

This project depends on several external packages. **Three submodules are included** in this codebase:

| Submodule | Description |
|---|---|
| `arx_mujoco/` | ARX R5 robot arm MuJoCo simulation & real-robot interface |
| `Hand2Gripper_model/` | The Hand-Gripper mapping neural network (training & inference) |
| `Hand2Gripper_vslam/` | Visual SLAM for camera trajectory estimation |

**Nine additional submodules** must be downloaded separately — see [`submodules/README.md`](submodules/README.md).
