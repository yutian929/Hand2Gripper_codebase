# Submodules — Download & Setup Guide

Due to **file-size constraints** and **double-blind review policy**, several submodules are not included in this supplementary material. Reviewers should download and place them manually following the instructions below.

> **Note:** Submodules marked with ★ are our contributions/modifications. They will be released as open-source repositories upon paper acceptance. For review purposes, we describe the base versions and the modifications we made.

---

## Included Submodules (no action needed)

The following submodules are already included in this codebase:

| Directory | Description |
|---|---|
| `arx_mujoco/` | ARX R5 robot arm MuJoCo simulation interface and real-robot control |
| `Hand2Gripper_model/` | Hand-Gripper mapping neural network (training & inference) |
| `Hand2Gripper_vslam/` | Visual SLAM module for camera trajectory estimation (RTAB-Map client) |

---

## Required Submodules (must be downloaded)

Please download each submodule and place it in the `submodules/` directory with the **exact folder name** specified below.

### 1. SAM2 — Segment Anything Model 2

- **Folder name:** `sam2/`
- **Source:** https://github.com/facebookresearch/sam2
- **Commit:** `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- **Purpose:** Used for hand and arm segmentation in video sequences.

```bash
cd submodules/
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

### 2. ★ phantom-hamer — Modified HaMeR (Hand Mesh Recovery)

- **Folder name:** `phantom-hamer/`
- **Base project:** HaMeR — https://github.com/geopavlakos/hamer
- **Purpose:** 2D/3D hand pose and mesh estimation from RGB images. We use the modified version from the Phantom project (Lepert et al., CoRL 2025) which integrates ViTPose for whole-body keypoint detection.
- **Modifications:** Integration with the processing pipeline; batch inference support.
- **Setup:** Will be released upon acceptance. For review, please use the base HaMeR repository and follow its installation instructions. Place it as `submodules/phantom-hamer/`.

### 3. ★ Hand2Gripper_robosuite — Modified Robosuite

- **Folder name:** `Hand2Gripper_robosuite/`
- **Base project:** robosuite — https://github.com/ARISE-Initiative/robosuite
- **Purpose:** MuJoCo-based robot simulation framework. Used for digital twin robot rendering and overlay generation.
- **Modifications:** Added custom robot models (ARX R5), custom environments, and camera configurations for our pipeline. Extended from the Phantom project's modifications.
- **Setup:** Will be released upon acceptance. For review, install the base robosuite (`pip install robosuite`) and ensure MuJoCo is properly configured.

### 4. ★ phantom-robomimic — Modified Robomimic

- **Folder name:** `phantom-robomimic/`
- **Base project:** robomimic — https://github.com/ARISE-Initiative/robomimic
- **Purpose:** Imitation learning data format and training utilities.
- **Modifications:** Custom data loading for our processed demonstration format.
- **Setup:** Will be released upon acceptance. For review, install the base robomimic (`pip install robomimic`).

### 5. ★ phantom-E2FGVI — Modified E2FGVI (Video Inpainting)

- **Folder name:** `phantom-E2FGVI/`
- **Base project:** E2FGVI — https://github.com/MCG-NKU/E2FGVI
- **Purpose:** Flow-guided video inpainting for removing human hands from demonstration videos, producing clean background frames for robot overlay rendering.
- **Modifications:** Adapted for batch processing with segmentation masks from our pipeline.
- **Setup:** Will be released upon acceptance. For review:
  1. Clone the base E2FGVI repository and place it as `submodules/phantom-E2FGVI/`
  2. Download the pre-trained model weights:
     ```bash
     cd submodules/phantom-E2FGVI/E2FGVI/release_model/
     pip install gdown
     gdown --fuzzy https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing
     ```

### 6. ★ Hand2Gripper_WiLoR-mini — Modified WiLoR

- **Folder name:** `Hand2Gripper_WiLoR-mini/`
- **Base project:** WiLoR — https://github.com/rolpotamern/WiLoR
- **Purpose:** Alternative hand pose estimation model (lighter/faster than HaMeR). Used when `hand_model: 'wilor'` is set in the config.
- **Modifications:** Minor API adaptations for integration with our processing pipeline.
- **Setup:** Will be released upon acceptance. For review, install the base WiLoR and place it as `submodules/Hand2Gripper_WiLoR-mini/`.

### 7. ★ Hand2Gripper_HACO — HACO Contact Estimator

- **Folder name:** `Hand2Gripper_HACO/`
- **Base project:** HACO — https://github.com/JudyYe/haco
- **Purpose:** Hand-object contact estimation. Predicts contact maps between hand meshes and manipulated objects, used to guide the gripper closure timing in the mapping model.
- **Modifications:** Integrated HaMeR-based hand mesh input; added a release checkpoint for our trained contact model.
- **Setup:** Will be released upon acceptance. For review:
  1. Clone the base HACO repository and place it as `submodules/Hand2Gripper_HACO/`
  2. Install dependencies: `pip install mediapipe easydict`
  3. Download pre-trained weights and place in `base_data/release_checkpoint/`

### 8. ★ Hand2Gripper_hand2gripper — Hand-Gripper Mapping Inference

- **Folder name:** `Hand2Gripper_hand2gripper/`
- **Purpose:** Standalone inference package for the Hand-Gripper mapping model (see also `Hand2Gripper_model/` for training code). Contains the release checkpoint used by `hand2gripper/hand.py`.
- **Setup:** Will be released upon acceptance. The trained checkpoint (`hand2gripper.pt`) is loaded from `submodules/Hand2Gripper_hand2gripper/hand2gripper/release_checkpoint/`.

---

## Quick Setup Summary

After downloading all submodules, your `submodules/` directory should look like:

```
submodules/
├── arx_mujoco/                 (included)
├── Hand2Gripper_model/         (included)
├── Hand2Gripper_vslam/         (included)
├── sam2/                       (download)
├── phantom-hamer/              (download)
├── Hand2Gripper_robosuite/     (download)
├── phantom-robomimic/          (download)
├── phantom-E2FGVI/             (download)
├── Hand2Gripper_WiLoR-mini/    (download)
├── Hand2Gripper_HACO/          (download)
└── Hand2Gripper_hand2gripper/  (download)
```

Then run `bash install.sh` from the project root to install everything.