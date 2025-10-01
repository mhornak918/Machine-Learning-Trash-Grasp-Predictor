# Trash Segmentation and Grasp Estimation

A deep learning-based system for estimating robotic grasps on trash objects using RGB images, enabling two-finger robotic grippers to accurately grasp waste items in unconstrained environments.

## Overview

This project bridges the gap between trash detection and robotic grasping by combining the TACO trash segmentation dataset with the Jacquard grasping dataset. We compare two approaches:
- **RGB Model**: Uses only RGB images and segmentation masks
- **RGBD Model**: Incorporates synthetic depth maps generated using MiDaS

## Key Features

- Custom trash segmentation using Mask R-CNN trained on TACO dataset
- Domain adaptation pipeline to apply Jacquard grasp annotations to real-world trash
- Synthetic depth generation using MiDaS for depth-aware grasping without physical depth sensors
- Centroid-based grasp localization for improved stability and accuracy

## Results

| Model | Success Rate |
|-------|--------------|
| RGBD (with MiDaS depth) | **90.81%** |
| RGB (no depth) | **83.64%** |

The RGBD model achieved higher accuracy despite being trained on 40% less data, demonstrating the value of synthetic depth information for grasp estimation.

## Architecture

### Segmentation Pipeline
- **Model**: Mask R-CNN with ResNet-50 backbone and Feature Pyramid Network
- **Dataset**: TACO (1,500 images, 4,784 labeled trash instances, 60 categories)
- **Training**: Transfer learning from COCO pretrained weights
- **Output**: Binary segmentation masks for downstream grasping

### Grasp Estimation Pipeline
- **Base Architecture**: Modified ResNet-18
- **Input Channels**: 
  - RGB: 4 channels (RGB + mask)
  - RGBD: 5 channels (RGB + depth + mask)
- **Output**: Grasp angle (θ) and gripper width (w)
- **Grasp Location**: Computed from segmentation mask centroid
- **Training Dataset**: Jacquard (3,000-4,500 unique objects)

## Methodology

1. **Segmentation**: Detect and isolate trash objects using Mask R-CNN
2. **Depth Estimation** (RGBD only): Generate synthetic depth maps using MiDaS DPT-Large
3. **Preprocessing**: 
   - Crop around segmented object with symmetric scaling
   - Resize to 224×224
   - Normalize depth to Jacquard range [1.39, 1.57] meters
   - Apply gray background masking
4. **Grasp Prediction**: Estimate grasp angle and width using trained model
5. **Visualization**: Render grasp rectangle with gripper fingers (red) and opening width (green)

## Key Design Decisions

### Centroid-Based Localization
Rather than predicting grasp location (x, y) directly, we compute it from the segmentation mask centroid. This approach:
- Reduces output dimensionality and training complexity
- Eliminates noisy gradients from ambiguous positioning
- Improves generalization across varied object appearances
- Achieved 90.81% success vs 41.82% when predicting all parameters

### Synthetic Depth Integration
Using MiDaS depth estimation allows:
- Processing RGB-only trash datasets in depth-aware frameworks
- Avoiding expensive RGBD sensor requirements
- Providing critical geometric cues for grasp orientation
- Better performance with less training data (40% reduction)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- albumentations
- pycocotools
- MiDaS (for depth estimation)
- OpenCV
- NumPy

## Training Details

### Segmentation Model
- **Epochs**: 30 (10 frozen backbone + 20 fine-tuning)
- **Optimizer**: AdamW
- **Learning Rate**: 1×10⁻³ (stage 1), 1×10⁻⁴ (stage 2)
- **Loss**: Multi-task (cross-entropy, Smooth L1, binary cross-entropy)
- **Hardware**: Google Colab T4 GPU (~5 hours total)

### Grasp Models
- **Epochs**: 10 initial + 5 fine-tuning
- **Optimizer**: Adam
- **Learning Rate**: 1×10⁻⁴ (initial), 1×10⁻⁵ (fine-tuning)
- **Batch Size**: 16
- **Loss**: Smooth L1 (weakly supervised)
- **Augmentation**: Color jitter, Gaussian noise (σ=0.01)

## Datasets

- **TACO**: Trash Annotations in Context - [tacodataset.org](https://tacodataset.org)
- **Jacquard**: Robotic grasping dataset with RGBD and grasp annotations

## Limitations and Future Work

### Current Limitations
- Small TACO dataset size limits object diversity
- Severe class imbalance (rare categories underperform)
- Many artificial scenes may not reflect real-world distributions
- Complex occlusion and clutter remain challenging
- Only evaluated on offline predictions (no physical robot testing)

### Future Directions
- Collect larger labeled dataset in realistic waste environments
- Generate synthetic training data using photorealistic rendering
- Explore advanced segmentation (YOLOv11, Segment Anything Model)
- Train on full Jacquard dataset for improved robustness
- Conduct real-world robotic validation with physical grasping
- Implement end-to-end joint training of segmentation and grasping
- Incorporate multi-view or temporal visual data

## Team

- **Batyrkhan Baimukhanov** - Trash segmentation model (TACO)
- **Mitchell Hornak** - RGBD grasp estimation model (Jacquard + MiDaS)
- **Siddarth Bhupathiraju** - RGB grasp estimation model (Jacquard)

Boston University

## References

1. Lenz et al. "Deep learning for detecting robotic grasps." IJRR, 2015
2. Li & Yuan. "Jacquard v2: Refining datasets using human in the loop." ICRA, 2024
3. Proença & Simões. "TACO: Trash annotations in context." JFR, 2020
4. Ranftl et al. "Towards robust monocular depth estimation." TPAMI, 2020
5. Redmon & Angelova. "Real-time grasp detection using CNNs." ICRA, 2015

## Citation

If you use this work, please cite:
```
Baimukhanov, B., Bhupathiraju, S., & Hornak, M. (2025).
Trash Segmentation and Grasp Estimation: A Comparative Study of RGB vs. RGBD 
Neural Networks for Mask-Guided Robotic Grasping. Boston University.
```

## License

[Specify your license here]
