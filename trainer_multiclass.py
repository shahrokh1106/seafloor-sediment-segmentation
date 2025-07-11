import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import JaccardIndex
from collections import Counter
import argparse


class DINOv2SegHead(nn.Module):
    def __init__(self, backbone_name='vit_base_patch14_dinov2.lvd142m', num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.num_levels = 3
        in_channels_per_level = 768 
        projection_dim = 128
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_per_level, projection_dim, kernel_size=1),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels)
        ])
        total_decoder_in_channels = projection_dim * self.num_levels  # = 128 * 3 = 384
        self.decoder = nn.Sequential(
            nn.Conv2d(total_decoder_in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.upsampler = JBUStack(feat_dim=num_classes)
 
    def forward(self, x):
        features = self.backbone(x)
        projected_feats = [proj(feat) for feat, proj in zip(features, self.proj_layers)]
        fused = torch.cat(projected_feats, dim=1)  
        out = self.decoder(fused)
        return  F.interpolate(self.upsampler(out,x), size=(518, 518), mode='bilinear', align_corners=False) 

class DINOv2SegHeadV2(nn.Module):
    def __init__(self, backbone_name='vit_base_patch14_dinov2.lvd142m', num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.num_levels = 3
        in_channels_per_level = 768 
        projection_dim = 128
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_per_level, projection_dim, kernel_size=1),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels)
        ])
        total_decoder_in_channels = projection_dim * self.num_levels  # = 128 * 3 = 384
        self.decoder = nn.Sequential(
            nn.Conv2d(total_decoder_in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False)

    def forward(self, x):
        features = self.backbone(x)
        projected_feats = [proj(feat) for feat, proj in zip(features, self.proj_layers)]
        fused = torch.cat(projected_feats, dim=1)  # Shape: [B, 384, 37, 37]
        out = self.decoder(fused)
        return self.upsample(out)
    
# ---------------------------
# DATASET
# ---------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)/30

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = torch.tensor(augmented['mask'], dtype=torch.long)  # shape: [H, W]
        return image, mask

def get_transforms(input_size):
    train_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform


# ---------------------------
# TRAINING
# ---------------------------
class SegmentationTrainer:
    def __init__(self, model,input_size, model_name, train_loader, val_loader, device,
                 initial_lr=1e-4, patience=10, output_path="", freeze=True, num_classes=4,
                 weight_ce=1.0, weight_dice=1.0,class_weights= None):
        self.model = model.to(device)

        self.input_size = input_size
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_path = output_path
        self.freeze = freeze
        self.patience = patience
        self.initial_lr = initial_lr
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.class_weights = class_weights
        if class_weights != None:
            self.class_weights = class_weights.to(device)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.freeze:
            if "dino" not in self.model_name:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.backbone.parameters():
                    param.requires_grad = False

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.initial_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.iou_metric = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

        self.best_iou = 0
        self.early_stop_counter = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_iou_history = []
        self.val_iou_history = []

    def multiclass_dice_loss(self, preds, targets):
        # preds: [B, C, H, W], targets: [B, H, W]
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (preds * targets_one_hot).sum(dims)
        union = preds.sum(dims) + targets_one_hot.sum(dims)
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        return 1 - dice.mean()

    def combined_loss(self, preds, targets):
        ce = self.ce_loss(preds, targets)
        dice = self.multiclass_dice_loss(preds, targets)
        return self.weight_ce * ce + self.weight_dice * dice

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_iou = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch [{epoch}] - Training", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)  # [B, C, H, W]
            loss = self.combined_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou = self.iou_metric(preds, labels)
            all_iou.append(iou.item())

        epoch_loss = running_loss / len(self.train_loader)
        epoch_iou = np.mean(all_iou)
        self.train_loss_history.append(epoch_loss)
        self.train_iou_history.append(epoch_iou)
        return epoch_loss, epoch_iou

    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_iou = []
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch [{epoch}] - Validation", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.combined_loss(outputs, labels)
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                iou = self.iou_metric(preds, labels)
                all_iou.append(iou.item())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_iou = np.mean(all_iou)
        self.val_loss_history.append(epoch_loss)
        self.val_iou_history.append(epoch_iou)
        self.scheduler.step(epoch_iou)
        return epoch_loss, epoch_iou

    def fit(self, epochs):
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            train_loss, train_iou = self.train_one_epoch(epoch)
            val_loss, val_iou = self.validate_one_epoch(epoch)
            print(f"\n[Epoch {epoch}] Training - Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
            print(f"[Epoch {epoch}] Validation - Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.early_stop_counter = 0
                print(f"[INFO] New best model found! Saving...")
                torch.save(self.model.state_dict(), os.path.join(self.output_path, self.model_name + ".pth"))
            else:
                self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                break
def compute_class_weights(mask_paths, num_classes):
    class_counts = Counter()
    print("Computing Class Weights ...")
    for path in tqdm(mask_paths):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = (mask // 30).astype(np.uint8)

        unique, counts = np.unique(mask, return_counts=True)
        class_counts.update(dict(zip(unique, counts)))
    total = sum(class_counts.values())
    freqs = np.array([class_counts.get(i, 0) / total for i in range(num_classes)])
    weights = 1.0 / (freqs + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize to sum ≈ num_classes
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
     
    NUM_CLASSES =  6+1
    model = DINOv2SegHeadV2(num_classes=NUM_CLASSES)
    input_size = 518
    batch_size = 4
    parser = argparse.ArgumentParser(description="Binary segmentation training")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("output_path", type=str, help="Path to save the model")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    output_path = os.path.join(output_path, "multiclass")
    os.makedirs(output_path,exist_ok=True) 
    image_dir = os.path.join(dataset_path, "images")
    mask_dir = os.path.join(dataset_path, "labels")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))+sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))+sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    train_transform, val_transform = get_transforms(input_size)

    class_weights = compute_class_weights(train_masks, num_classes=NUM_CLASSES)
    # class_weights = None
    print("Number of images:", len(image_paths))
    print("Number of masks:", len(mask_paths))

    train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = SegmentationDataset(val_imgs, val_masks, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SegmentationTrainer(
        model=model,
        input_size = input_size,
        model_name="DINOv2SegHeadV2DatasetV0",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_path=os.path.join(output_path),
        freeze=False,
        initial_lr=1e-4,
        num_classes=NUM_CLASSES,            
        weight_ce=0.5,
        weight_dice=1.5,
        class_weights = class_weights
        )

    trainer.fit(epochs=70)
