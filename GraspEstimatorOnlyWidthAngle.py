import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
import glob
import torch.nn as nn
from tqdm import tqdm
import math
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
import argparse

def safe_open(path):
    if not os.path.exists(path):
        print(f"Missing: {path}")
    return Image.open(path)

class JacquardDataset(Dataset):
    def __init__(self, folder_list, base_path, transform=None, use_stereo_depth=False):
        self.base_path = base_path
        self.transform = transform
        self.use_stereo_depth = use_stereo_depth
        self.samples = []

        for folder in folder_list:
            folder_path = os.path.join(self.base_path, folder)

            # Find all RGB files and extract unique prefixes like "0_", "1_", etc.
            rgb_files = glob.glob(os.path.join(folder_path, "*RGB.png"))
            prefixes = set(os.path.basename(f).split("_RGB")[0] for f in rgb_files)

            for prefix in prefixes:
                rgb_path = os.path.join(folder_path, f"{prefix}_RGB.png")
                depth_name = f"{prefix}_stereo_depth.tiff" if use_stereo_depth else f"{prefix}_perfect_depth.tiff"
                depth_path = os.path.join(folder_path, depth_name)
                grasp_path = os.path.join(folder_path, f"{prefix}_grasps.txt")
                mask_path = os.path.join(folder_path, f"{prefix}_mask.png")

                if all(os.path.exists(p) for p in [rgb_path, depth_path, grasp_path, mask_path]):
                    self.samples.append({
                        "rgb": rgb_path,
                        "depth": depth_path,
                        "grasps": grasp_path,
                        "mask": mask_path
                    })

        print(f"Total valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb = Image.open(sample["rgb"]).convert("RGB")
        depth = Image.open(sample["depth"])
        mask = Image.open(sample["mask"]).convert("L")

        if self.transform:
            rgbd_mask = self.transform(rgb, depth, mask)
        else:
            rgb_tensor = transforms.ToTensor()(rgb)
            depth_tensor = transforms.ToTensor()(depth)
            mask_tensor = transforms.ToTensor()(mask)
            rgbd_mask = torch.cat((rgb_tensor, depth_tensor, mask_tensor), dim=0)

        with open(sample["grasps"], 'r') as f:
            grasps = [list(map(float, line.strip().split(';'))) for line in f if line.strip()]
        grasps = np.array(grasps)

        # RESCALE TO 224×224
        orig_w, orig_h = rgb.size
        scale_x = 224 / orig_w
        scale_y = 224 / orig_h
        grasps[:, 0] *= scale_x
        grasps[:, 1] *= scale_y
        grasps[:, 3] *= (scale_x + scale_y) / 2  # grasp width

        return rgbd_mask, grasps

class GraspPredictorNet(nn.Module):
    def __init__(self):
        super(GraspPredictorNet, self).__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)

def custom_collate_fn(batch):
    rgbd_batch = torch.stack([item[0] for item in batch], dim=0)
    grasps_batch = [item[1] for item in batch]
    return rgbd_batch, grasps_batch

def split_dataset(base_path, val_ratio=0.1):
    all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    random.shuffle(all_folders)
    n_val = int(len(all_folders) * val_ratio)
    return all_folders[n_val:], all_folders[:n_val]

def draw_grasp_on_rgb(rgb_tensor, grasp, ax=None, color=(0, 255, 0)):
    x, y, theta, width, quality = grasp
    x = np.clip(x, 0, 223)
    y = np.clip(y, 0, 223)
    width = np.clip(width, 5, 150)
    theta = np.clip(theta, -180, 180)
    h = 20
    theta_rad = math.radians(theta) if abs(theta) > 2 * math.pi else theta
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    dx = (width / 2) * cos_t
    dy = (width / 2) * sin_t
    p1 = (x - dx - h * sin_t, y - dy + h * cos_t)
    p2 = (x + dx - h * sin_t, y + dy + h * cos_t)
    p3 = (x + dx + h * sin_t, y + dy - h * cos_t)
    p4 = (x - dx + h * sin_t, y - dy - h * cos_t)
    box = np.array([p1, p2, p3, p4], dtype=np.int32)
    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_np = (rgb_np * 255).astype(np.uint8)
    rgb_with_grasp = cv2.polylines(rgb_np.copy(), [box], isClosed=True, color=color, thickness=2)
    if ax is None:
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb_with_grasp)
        plt.title(f"Predicted Grasp (q={quality:.2f})")
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb_with_grasp)
        ax.set_title(f"Predicted Grasp (q={quality:.2f})")
        ax.axis('off')

from torchvision import transforms
import torch

# --- NEW Augmentation Class ---
class AugmentRGBOnly:
    def __init__(self, add_noise=True, noise_std=0.01):
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])
        self.gray_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __call__(self, rgb_img, depth_img, mask_img):
        rgb_tensor = self.rgb_transform(rgb_img)
        if self.add_noise:
            noise = torch.randn_like(rgb_tensor) * self.noise_std
            rgb_tensor = torch.clamp(rgb_tensor + noise, 0.0, 1.0)

        depth_tensor = self.gray_transform(depth_img)
        mask_tensor = self.gray_transform(mask_img)
        return torch.cat([rgb_tensor, depth_tensor, mask_tensor], dim=0)

def train_model(train_loader, val_loader, model, device, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for rgbd_batch, grasps_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            rgbd_batch = rgbd_batch.to(device)
            preds = model(rgbd_batch)  # shape: (B, 5)

            batch_loss = 0.0
            for i in range(len(preds)):
                gt = torch.tensor(grasps_batch[i], device=preds.device)[:, [2, 3]]  # only angle and width
                pred = preds[i].unsqueeze(0)  # (1, 5), for broadcasting

                # Compute Smooth L1 loss to all grasps in this sample
                all_losses = torch.nn.functional.smooth_l1_loss(
                    gt, pred.expand_as(gt), reduction='none'
                ).mean(dim=1)  # shape: (N_grasps,)
                min_loss = all_losses.min()
                batch_loss += min_loss

            batch_loss = batch_loss / len(preds)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgbd_batch, grasps_batch in val_loader:
                rgbd_batch = rgbd_batch.to(device)
                preds = model(rgbd_batch)

                batch_loss = 0.0
                for i in range(len(preds)):
                    gt = torch.tensor(grasps_batch[i], device=preds.device)[:, [2, 3]]
                    pred = preds[i].unsqueeze(0)

                    all_losses = torch.nn.functional.smooth_l1_loss(
                        gt, pred.expand_as(gt), reduction='none'
                    ).mean(dim=1)
                    min_loss = all_losses.min()
                    batch_loss += min_loss

                val_loss += batch_loss.item() / len(preds)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}\n")

    return model

def verify_sample(dataset):
    rgbd, grasps = dataset[0]
    rgb = rgbd[:3, :, :].permute(1, 2, 0).numpy()
    depth = rgbd[3, :, :].numpy()
    mask = rgbd[4, :, :].numpy()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("RGB"); plt.imshow(rgb)
    plt.subplot(1, 3, 2); plt.title("Depth"); plt.imshow(depth, cmap='gray')
    plt.subplot(1, 3, 3); plt.title("Mask"); plt.imshow(mask, cmap='gray')
    plt.show()
    print("Grasps shape:", grasps.shape)
    print("Grasp samples:\n", grasps[:3])

def predict_and_visualize(model, dataset, device):
    model.eval()
    rgbd_tensor, _ = dataset[0]
    rgbd_tensor = rgbd_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(rgbd_tensor)[0].cpu().numpy()
    print(f"Prediction: x={pred[0]:.2f}, y={pred[1]:.2f}, angle={pred[2]:.2f}, width={pred[3]:.2f}, quality={pred[4]:.2f}")
    rgb_tensor = rgbd_tensor[0, :3, :, :]
    draw_grasp_on_rgb(rgb_tensor, pred)

def main(mode="train"):
    base_path = "C:\\Users\\mrhor\\Documents\\CS585\\Project\\Jacquard\\Jacquard_Dataset_11"
    train_folders, val_folders = split_dataset(base_path)
    transform_train = AugmentRGBOnly(add_noise=True, noise_std=0.01)
    transform_val = None
    train_dataset = JacquardDataset(train_folders, base_path, transform=transform_train)
    val_dataset = JacquardDataset(val_folders, base_path, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraspPredictorNet().to(device)

    if mode == "verify":
        verify_sample(train_dataset)
    elif mode == "train":
        loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = train_model(train_loader, val_loader, model, device, loss_fn, optimizer, epochs=10)
        torch.save(model.state_dict(), "grasp_rgbd_mask_augmented_WidthAngleOnly.pth")
    elif mode == "predict":
        model.load_state_dict(torch.load("grasp_predictor_rgbd_mask.pth", map_location=device))
        predict_and_visualize(model, val_dataset, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="Options: train, verify, predict")
    args = parser.parse_args()
    main(mode=args.mode)