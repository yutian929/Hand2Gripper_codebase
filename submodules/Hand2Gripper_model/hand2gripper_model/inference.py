# inference.py
# -*- coding: utf-8 -*-
"""
Hand-to-Gripper model inference script.

Features:
- Load a trained Hand2GripperModel.
- Read a sample data file in .npz format.
- Execute the complete preprocessing, model inference, and postprocessing pipeline.
- Print the predicted results (left, right) and optionally generate a visualization image.

Usage example:
  # Perform inference on sample.npz and save the visualization to output.png
  python inference.py --checkpoint hand2gripper.pt --input sample.npz --output output.png

  # Print results only, without generating visualization
  python inference.py --checkpoint hand2gripper.pt --input sample.npz
"""
import os
import argparse
import numpy as np
import torch
import cv2
from typing import Dict

# Assumes this script is at the same level as the models directory
from .models.simple_pair import Hand2GripperModel

# ------------------------------
# Visualization utility functions
# ------------------------------
def vis_selected_gripper(image: np.ndarray, kpts_2d: np.ndarray, gripper_joints_pair: np.ndarray) -> np.ndarray:
    """
    Draw selected gripper joint points and connecting lines on the image.
    
    Args:
        image: Original image [H, W, 3]
        kpts_2d: 2D keypoints [21, 2]
        gripper_joints_pair: Predicted (left, right) indices [2]
    """
    img_vis = image.copy()
    colors = [(0, 0, 255), (255, 0, 0)]  # Left: Blue, Right: Red
    labels = ["L", "R"]
    
    # Draw connecting line (left -> right)
    left_pt = tuple(kpts_2d[gripper_joints_pair[0]].astype(int))
    right_pt = tuple(kpts_2d[gripper_joints_pair[1]].astype(int))
    cv2.line(img_vis, left_pt, right_pt, (255, 255, 0), 2)  # Cyan

    # Draw joint points
    for i, joint_id in enumerate(gripper_joints_pair):
        pt = tuple(kpts_2d[joint_id].astype(int))
        cv2.circle(img_vis, pt, 5, colors[i], -1)
        cv2.putText(img_vis, labels[i], (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        
    return img_vis

# ------------------------------
# Inference class
# ------------------------------
class Hand2GripperInference:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Initialize the model and load weights.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.device = torch.device(device)
        self.model = Hand2GripperModel(d_model=256, img_size=256)
        
        # Use the model's built-in load function
        self.model._load_checkpoint(checkpoint_path)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {checkpoint_path} and set to evaluation mode on {self.device}.")

    @torch.no_grad()
    def predict_t(self, color: np.ndarray, bbox: np.ndarray, keypoints_3d: np.ndarray,
                contact: np.ndarray, is_right: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Execute the complete inference pipeline on a single sample.

        Args:
            color (np.ndarray): Original image, [H,W,3] or [3,H,W], uint8/float
            bbox (np.ndarray): Bounding box, [4]
            keypoints_3d (np.ndarray): 3D keypoints, [21,3]
            contact (np.ndarray): Contact probabilities/logits, [21]
            is_right (np.ndarray): Whether this is the right hand, scalar or [1]

        Returns:
            Dict[str, torch.Tensor]: The raw output dictionary from the model.
        """
        # 1. Use the model's internal read functions to convert numpy arrays to batched tensors
        color_t = self.model._read_color(color).to(self.device)
        bbox_t = self.model._read_bbox(bbox).to(self.device)
        kp3d_t = self.model._read_keypoints_3d(keypoints_3d).to(self.device)
        contact_t = self.model._read_contact(contact).to(self.device)
        isright_t = self.model._read_is_right(is_right).to(self.device)

        # 2. Preprocessing: crop and resize the image
        # Note: this step is done outside the model, consistent with the training script
        crop_t = self.model._crop_and_resize(color_t, bbox_t)

        # 3. Model forward pass
        outputs = self.model(crop_t, kp3d_t, contact_t, isright_t)
        
        return outputs

    @torch.no_grad()
    def predict(self, color: np.ndarray, bbox: np.ndarray, keypoints_3d: np.ndarray,
                contact: np.ndarray, is_right: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Execute the complete inference pipeline on a single sample.

        Args:
            color (np.ndarray): Original image, [H,W,3] or [3,H,W], uint8/float
            bbox (np.ndarray): Bounding box, [4]
            keypoints_3d (np.ndarray): 3D keypoints, [21,3]
            contact (np.ndarray): Contact probabilities/logits, [21]
            is_right (np.ndarray): Whether this is the right hand, scalar or [1]

        Returns:
            np.ndarray: Predicted (left, right) keypoint indices [2]
        """
        # 1. Use the model's internal read functions to convert numpy arrays to batched tensors
        color_t = self.model._read_color(color).to(self.device)
        bbox_t = self.model._read_bbox(bbox).to(self.device)
        kp3d_t = self.model._read_keypoints_3d(keypoints_3d).to(self.device)
        contact_t = self.model._read_contact(contact).to(self.device)
        isright_t = self.model._read_is_right(is_right).to(self.device)

        # 2. Preprocessing: crop and resize the image
        # Note: this step is done outside the model, consistent with the training script
        crop_t = self.model._crop_and_resize(color_t, bbox_t)

        # 3. Model forward pass
        outputs = self.model(crop_t, kp3d_t, contact_t, isright_t)
        
        return outputs['pred_pair'].squeeze().cpu().numpy()
    
    def vis_output(self, image: np.ndarray, kpts_2d: np.ndarray, pred_pair: np.array) -> np.ndarray:
        """
        Visualize the model output results.

        Args:
            image (np.ndarray): Original image, [H,W,3], uint8
            kpts_2d (np.ndarray): 2D keypoints, [21,2]
            pred_pair (np.ndarray): Predicted gripper joint indices (left, right), [2]

        Returns:
            np.ndarray: Image with visualization results.
        """
        vis_img = vis_selected_gripper(image, kpts_2d, pred_pair)
        return vis_img

# ------------------------------
# Main function
# ------------------------------
def main(args):
    # Check input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input data file not found: {args.input}")

    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Initialize inference engine
    try:
        inference_engine = Hand2GripperInference(args.checkpoint, device=device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Load data
    print(f"Loading data from {args.input}...")
    try:
        data = np.load(args.input, allow_pickle=True)
        # Extract required data from .npz file
        color_np = data["img_rgb"]
        bbox_np = data["bbox"]
        kpts_3d_np = data["kpts_3d"]
        contact_np = data["contact_logits"]
        is_right_np = data["is_right"]
        kpts_2d_np = data["kpts_2d"]  # for visualization
    except Exception as e:
        print(f"Error loading data from {args.input}. Ensure it's a valid .npz file with required keys. Error: {e}")
        return

    # Run inference
    outputs = inference_engine.predict(
        color=color_np,
        bbox=bbox_np,
        keypoints_3d=kpts_3d_np,
        contact=contact_np,
        is_right=is_right_np
    )

    # Postprocess and print results
    pred_pair = outputs
    print("\n" + "="*30)
    print("      Inference Result")
    print("="*30)
    print(f"Predicted Gripper Pair (Left, Right): {pred_pair}")
    print(f"  - Left Joint ID:  {pred_pair[0]}")
    print(f"  - Right Joint ID: {pred_pair[1]}")
    print("="*30)

    # Visualization
    if args.output:
        print(f"\nGenerating visualization and saving to {args.output}...")
        # Ensure image is in HWC, uint8, BGR format for OpenCV
        if color_np.dtype != np.uint8:
            vis_img = (color_np * 255).astype(np.uint8)
        else:
            vis_img = color_np.copy()
        
        if vis_img.shape[0] == 3: # CHW -> HWC
            vis_img = np.transpose(vis_img, (1, 2, 0))
        
        # RGB -> BGR for OpenCV
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # Draw results
        vis_img_result = inference_engine.vis_output(vis_img, kpts_2d_np, pred_pair)
        
        # Save image
        try:
            cv2.imwrite(args.output, vis_img_result)
            print(f"Visualization saved successfully.")
        except Exception as e:
            print(f"Error saving visualization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand2Gripper Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data sample (.npz file).")
    parser.add_argument("--output", type=str, default=None, help="Optional: Path to save the output visualization image.")
    parser.add_argument("--cpu", action="store_true", help="Force use CPU even if CUDA is available.")
    
    args = parser.parse_args()
    main(args)