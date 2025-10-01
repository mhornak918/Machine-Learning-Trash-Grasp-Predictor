import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from GraspEstimatorwNoise import (
    JacquardDataset,
    GraspPredictorNet,
    custom_collate_fn,
    train_model,
    split_dataset
)
from GraspEstimatorwNoise import AugmentRGBOnly
import torch.nn as nn

# --- Config ---
BASE_PATH = "C:\\Users\\mrhor\\Documents\\CS585\\Project\\Jacquard\\Jacquard_Dataset_11"
BATCH_SIZE = 16
EPOCHS = 5
PRETRAINED_PATH = "grasp_rgbd_mask_augmented.pth"
OUTPUT_PATH = "grasp_rgbd_mask_finetuned_stereo_augmented.pth"

transform = AugmentRGBOnly(add_noise=True, noise_std=0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split folders and create datasets using stereo depth
train_folders, val_folders = split_dataset(BASE_PATH)
train_dataset = JacquardDataset(train_folders, BASE_PATH, transform=transform, use_stereo_depth=True)
val_dataset = JacquardDataset(val_folders, BASE_PATH, transform=None, use_stereo_depth=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# Load model from pretrained weights
model = GraspPredictorNet().to(device)
model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))

# Optionally freeze backbone if you only want to adapt FC layers
# for param in model.backbone.parameters():
#     param.requires_grad = False

# Fine-tune with a smaller learning rate
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train
model = train_model(train_loader, val_loader, model, device, loss_fn, optimizer, epochs=EPOCHS)

# Save fine-tuned model
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"Fine-tuned model saved to {OUTPUT_PATH}")
