import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import DPTForDepthEstimation, DPTImageProcessor
from GraspEstimatorwNoise import GraspPredictorNet, draw_grasp_on_rgb
import math

# --- Configuration ---
ROOT_DIR = "trashResults/masks"
MODEL_PATH = "grasp_rgbd_mask_finetuned_stereo_augmented.pth"
NUM_SAMPLES = 55
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Grasp Estimator ---
model = GraspPredictorNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Load MiDaS (transformers version) ---
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(DEVICE).eval()
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")

# --- Transformations ---
resize_224 = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()

# --- RGB + Mask -> RGBD Tensor ---
def prepare_rgbd(rgb_path, mask_path):
    from PIL import ImageOps
    import cv2
    import numpy as np
    import torch
    from torchvision import transforms

    # Load full-size RGB and mask
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    mask_np = np.array(mask)
    if np.max(mask_np) == 0:
        raise ValueError("Mask is empty")

    # Find object bounding box
    y_indices, x_indices = np.where(mask_np > 128)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Expand bounding box symmetrically by scale factor (e.g., 1.5x)
    box_w = x_max - x_min
    box_h = y_max - y_min
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    scale = 10
    new_w = int(box_w * scale)
    new_h = int(box_h * scale)

    x_min = max(cx - new_w // 2, 0)
    y_min = max(cy - new_h // 2, 0)
    x_max = min(cx + new_w // 2, mask_np.shape[1])
    y_max = min(cy + new_h // 2, mask_np.shape[0])

    # Crop all inputs to padded bounding box
    rgb_cropped = rgb.crop((x_min, y_min, x_max, y_max))
    mask_cropped = mask.crop((x_min, y_min, x_max, y_max))

    # --- Run MiDaS on original RGB ---
    inputs = image_processor(images=rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        raw_depth = depth_model(**inputs).predicted_depth.squeeze().cpu().numpy()

    raw_depth_resized = cv2.resize(raw_depth, rgb.size, interpolation=cv2.INTER_CUBIC)
    depth_full = Image.fromarray(raw_depth_resized.astype(np.float32))
    depth_cropped = depth_full.crop((x_min, y_min, x_max, y_max))

    # Pad cropped inputs to square
    def pad_to_square(img, fill):
        w, h = img.size
        max_dim = max(w, h)
        delta_w = max_dim - w
        delta_h = max_dim - h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        return ImageOps.expand(img, padding, fill=fill)

    rgb_square = pad_to_square(rgb_cropped, fill=(255, 255, 255))
    mask_square = pad_to_square(mask_cropped, fill=0)
    depth_square = pad_to_square(depth_cropped, fill=1.58)

    # Resize all to 224×224
    resize = transforms.Resize((224, 224))
    rgb_resized = resize(rgb_square)
    mask_resized = resize(mask_square)
    depth_resized_np = cv2.resize(np.array(depth_square), (224, 224), interpolation=cv2.INTER_CUBIC)

    # Generate binary mask
    mask_np = np.array(mask_resized)
    binary_mask = mask_np > 128

    # Rescale depth to Jacquard range [1.39, 1.57]
    depth_norm = (depth_resized_np - depth_resized_np.min()) / (depth_resized_np.max() - depth_resized_np.min())
    depth_jacquard_scaled = 1.39 + depth_norm * (1.57 - 1.39)
    depth_jacquard_scaled[~binary_mask] = 1.58  # background depth

    # Apply white background to RGB
    rgb_np = np.array(rgb_resized)
    rgb_np[~binary_mask] = [180, 180, 180]

    # Convert to tensors
    to_tensor = transforms.ToTensor()
    rgb_tensor = to_tensor(rgb_np)  # (3, 224, 224)
    depth_tensor = torch.tensor(depth_jacquard_scaled).unsqueeze(0).float()  # (1, 224, 224)
    mask_tensor = to_tensor(mask_resized)  # (1, 224, 224)

    return torch.cat([rgb_tensor, depth_tensor, mask_tensor], dim=0)  # (5, 224, 224)

def debug_rgbd_tensor(rgbd_tensor):
    """
    Visualize the RGB, depth, and mask channels from a 5-channel input tensor.
    """
    if rgbd_tensor.dim() == 4:
        rgbd_tensor = rgbd_tensor[0]  # Remove batch dimension (1, 5, 224, 224) → (5, 224, 224)

    rgb = rgbd_tensor[:3].permute(1, 2, 0).cpu().numpy()
    depth = rgbd_tensor[3].cpu().numpy()
    mask = rgbd_tensor[4].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[1].imshow(depth, cmap="inferno")
    axs[1].set_title("Depth")
    axs[2].imshow(mask, cmap="gray")
    axs[2].set_title("Mask")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def draw_colored_grasps(rgb_tensor, pred_grasp):
    def get_grasp_rect_edges(x, y, theta_deg, width, height=20):
        theta = math.radians(theta_deg)
        dx = width / 2
        dy = height / 2

        # Rotation matrix
        R = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ])

        # Rectangle corners relative to center
        corners = np.array([
            [-dx, -dy],  # top-left
            [ dx, -dy],  # top-right
            [ dx,  dy],  # bottom-right
            [-dx,  dy]   # bottom-left
        ])

        # Rotate and shift to (x, y)
        rotated = np.dot(corners, R.T)
        translated = rotated + np.array([x, y])
        return np.int32(translated)

    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_np = (rgb_np * 255).astype(np.uint8).copy()

    # --- Predicted grasp (green width, red height)
    x, y, theta, width, quality = pred_grasp
    x = np.clip(x, 0, 223)
    y = np.clip(y, 0, 223)
    width = np.clip(width, 5, 150)
    pred_box = get_grasp_rect_edges(x, y, theta, width)

    # Draw predicted: width edges (green), height edges (red)
    cv2.line(rgb_np, tuple(pred_box[0]), tuple(pred_box[1]), (0, 255, 0), 2)  # top width
    cv2.line(rgb_np, tuple(pred_box[2]), tuple(pred_box[3]), (0, 255, 0), 2)  # bottom width
    cv2.line(rgb_np, tuple(pred_box[1]), tuple(pred_box[2]), (255, 0, 0), 2)  # right height
    cv2.line(rgb_np, tuple(pred_box[3]), tuple(pred_box[0]), (255, 0, 0), 2)  # left height

    # --- Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_np)
    plt.title("Prediction vs Ground Truth")
    plt.axis('off')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='green', label='Predicted Width'),
        Patch(facecolor='none', edgecolor='red', label='Predicted Height'),
        Patch(facecolor='none', edgecolor='blue', label='Ground Truth')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.show()

# --- Evaluate ---
good_count = 0
for i in range(1, NUM_SAMPLES + 1):
    folder = f"{i:04d}"
    rgb_path = os.path.join(ROOT_DIR, folder, f"{folder}.png")
    mask_path = os.path.join(ROOT_DIR, folder, f"{folder}_mask.png")

    if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
        print(f"[SKIP] Missing files in {folder}")
        continue

    try:
        rgbd_tensor = prepare_rgbd(rgb_path, mask_path).unsqueeze(0).to(DEVICE)
        debug_rgbd_tensor(rgbd_tensor.cpu())
    except Exception as e:
        print(f"[ERROR] {folder}: {e}")
        continue

    with torch.no_grad():
        pred = model(rgbd_tensor)[0].cpu().numpy()

    # --- Visualization ---
    rgb_tensor = rgbd_tensor[0, :3, :, :]
    draw_colored_grasps(rgb_tensor, pred)

    print(f"Sample {i} - Predicted grasp:")
    print(f"  x={pred[0]:.2f}, y={pred[1]:.2f}, angle={pred[2]:.2f}, width={pred[3]:.2f}, quality={pred[4]:.2f}")

    while True:
        key = input("Is this a good grasp? (y/n): ").strip().lower()
        if key == "y":
            good_count += 1
            break
        elif key == "n":
            break
        else:
            print("Please type 'y' or 'n'.")

    plt.close()

import matplotlib.pyplot as plt

# --- Final Score ---
accuracy = 100 * good_count / NUM_SAMPLES
print(f"\nFinal Manual Accuracy: {good_count}/{NUM_SAMPLES} = {accuracy:.2f}%")
