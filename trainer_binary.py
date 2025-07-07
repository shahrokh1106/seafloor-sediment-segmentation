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
import argparse
import random 

SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

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
        self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False)
    def forward(self, x):
        features = self.backbone(x)
        projected_feats = [proj(feat) for feat, proj in zip(features, self.proj_layers)]
        fused = torch.cat(projected_feats, dim=1)  # Shape: [B, 384, 37, 37]
        out = self.decoder(fused)
        return self.upsample(out)


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
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0) / 255.0  # Normalize binary mask to [0, 1]
        return image, mask
    

def get_transforms(input_size):
    train_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform



class SegmentationTrainer:
    def __init__(self, model, model_name, train_loader, val_loader, device,
                 initial_lr=1e-4, patience=10, output_path="", use_dice_loss=True,freeze = True):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.initial_lr = initial_lr
        self.output_path = output_path
        self.use_dice_loss = use_dice_loss
        self.freeze = freeze
        os.makedirs(output_path,exist_ok=True)
        self.bce = nn.BCEWithLogitsLoss()
        if self.freeze:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.initial_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)

        self.iou_metric = BinaryJaccardIndex().to(device)
        self.best_iou = 0
        self.patience = patience
        self.early_stop_counter = 0

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_iou_history = []
        self.val_iou_history = []

    def dice_loss(self, pred, target, smooth=1):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self,inputs, targets, alpha=0.8, gamma=2):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss
        return F_loss.mean()
    
    def combined_loss(self, pred, target):
        dice = self.dice_loss(pred, target) if self.use_dice_loss else 0
        focal =  self.focal_loss(pred, target)
        return focal + dice

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_iou = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch [{epoch}] - Training", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.combined_loss(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            iou = self.iou_metric(preds.int().squeeze(1), labels.int().squeeze(1))
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
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.combined_loss(outputs, labels)
                running_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                iou = self.iou_metric(preds.int().squeeze(1), labels.int().squeeze(1))
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
                torch.save(self.model.state_dict(), os.path.join(self.output_path, self.model_name+".pth"))
            else:
                self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                break

if __name__ == "__main__":
    input_size = 518
    parser = argparse.ArgumentParser(description="Binary segmentation training")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("output_path", type=str, help="Path to save the model")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    class_name = os.path.basename(dataset_path)
    output_path = args.output_path
    output_path = os.path.join(output_path, "binary", class_name)
    os.makedirs(output_path,exist_ok=True)

    image_dir = os.path.join(dataset_path, "images")
    mask_dir = os.path.join(dataset_path, "labels")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    print("number of images: ", len(image_paths))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    print("number of masks: ", len(mask_paths))

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=SEED)
    train_transform, val_transform = get_transforms(input_size=input_size)
    train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = SegmentationDataset(val_imgs, val_masks, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = DINOv2SegHead()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SegmentationTrainer(
        model=model,
        model_name="dinov2_segmentor",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_path=output_path,
        use_dice_loss=True,
        freeze = True,
        initial_lr=1e-4
    )
    # Train
    trainer.fit(epochs=45)