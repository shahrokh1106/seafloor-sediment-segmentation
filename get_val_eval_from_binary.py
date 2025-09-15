import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import json
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy
from sklearn.metrics import confusion_matrix
import itertools

MASK_SCALE = 30
SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

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
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) //MASK_SCALE

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

@torch.no_grad()
def cache_probs(models, loader, device):
    probs_all, gts_all = [], []
    for imgs, gt in loader:
        batch_probs = []
        for m in tqdm(models):
            m.eval()
            logits = m(imgs.to(device))          # (B,1,H,W) pre-sigmoid
            batch_probs.append(torch.sigmoid(logits).cpu())  # -> probs
        probs_all.append(torch.cat(batch_probs, dim=1))      # (B,6,H,W)
        gts_all.append(gt.cpu()) # (B,H,W)   
    probs6 = torch.cat(probs_all, dim=0)      # (N,6,H,W)
    gts    = torch.cat(gts_all,  dim=0)       # (N,H,W)
    return probs6, gts

def ordered_overlay(probs6, order, thr=0.5):
    """
    probs6: (N,6,H,W) torch.float
    order:  tuple/list of class indices in {0..5}, high priority -> low
    thr:    scalar or per-class array-like of length 6
    returns: (N,H,W) long in {0..6}, where 0=background, 1..6=classes
    """
    N, C, H, W = probs6.shape
    final   = torch.zeros((N,H,W), dtype=torch.long)
    claimed = torch.zeros((N,1,H,W), dtype=torch.bool)
    thr_vec = torch.tensor(thr).view(1,C,1,1) if np.ndim(thr)==1 else thr
    for c in order:
        pc = probs6[:, c:c+1, :, :]                  # (N,1,H,W)
        mask = (pc >= (0.5 if np.ndim(thr)==0 else thr_vec[:,c:c+1])) & (~claimed)
        final = torch.where(mask.squeeze(1), torch.tensor(c+1, dtype=torch.long), final)
        claimed |= mask
    return final

class MetricBundle:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.jaccard = MulticlassJaccardIndex(num_classes=num_classes, average=None).to(device)
        self.dice    = MulticlassF1Score   (num_classes=num_classes, average=None).to(device)
        self.dice_m  = MulticlassF1Score   (num_classes=num_classes, average="macro").to(device)
        self.acc     = MulticlassAccuracy  (num_classes=num_classes, average="micro").to(device)
        self.acc_m   = MulticlassAccuracy  (num_classes=num_classes, average="macro").to(device)

def compute_metrics_torchmetrics(preds, gts, mb: MetricBundle, label_map=None):
    """
    preds, gts: (N,H,W) long in {0..num_classes-1}
    Returns dict exactly like your schema.
    """
    for m in [mb.jaccard, mb.dice, mb.dice_m, mb.acc, mb.acc_m]:
        m.reset()

    NUM_CLASSES = mb.num_classes
    device = mb.device

    preds_flat  = preds.view(-1).to(device)
    labels_flat = gts.view(-1).to(device)

    per_class_iou  = mb.jaccard(preds_flat, labels_flat).detach().cpu().tolist()
    per_class_dice = mb.dice   (preds_flat, labels_flat).detach().cpu().tolist()
    mean_dice      = mb.dice_m (preds_flat, labels_flat).item()
    pix_acc        = mb.acc    (preds_flat, labels_flat).item()
    mean_class_acc = mb.acc_m  (preds_flat, labels_flat).item()

    cm = confusion_matrix(labels_flat.detach().cpu().numpy(),
                          preds_flat.detach().cpu().numpy(),
                          labels=list(range(NUM_CLASSES)))
    correct_per_class = cm.diagonal()
    total_per_class   = cm.sum(axis=1)
    pa_per_class = (correct_per_class / np.maximum(total_per_class, 1)).tolist()
    mean_pa = float(np.mean([pa for pa in pa_per_class if not np.isnan(pa)]))

    results_dict = {
        "label_map": label_map if label_map is not None else {i: i for i in range(NUM_CLASSES)},
        "mean_iou": float(np.mean(per_class_iou)),
        "per_category_iou": per_class_iou,
        "per_category_dice": per_class_dice,
        "mean_dice": mean_dice,
        "pixel_accuracy": float(pix_acc),
        "mean_class_accuracy": float(mean_class_acc),
        "pixel_accuracy_per_class": pa_per_class,
        "mean_pixel_accuracy": mean_pa,
        "confusion_matrix": cm.tolist(),
    }
    return results_dict

if __name__ == "__main__":
    label_map  = {0:"Background", 1:"Mud", 2:"Sand", 3:"Shellhash coverage", 4:"Dog Cockle Bed", 5:"PatchesWorms", 6:"Bryozoans"}
    input_size = 518
    binary_model_name = "dinov2_binary_segmentor.pth"
    class_names = [label_map[i] for i in range(1,7)]
    model_paths_dict = {}
    for index, name in enumerate(class_names):
        model_paths_dict.update({name: os.path.join("trained_models", "binary", name, binary_model_name)})
    class_names = list(model_paths_dict.keys())
    models_dict = dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for class_name in class_names:
        temp_model = DINOv2SegHead()
        temp_model.load_state_dict(torch.load(model_paths_dict[class_name], map_location=device))
        temp_model.eval()
        temp_model.to(device)
        models_dict.update({class_name:temp_model})
        print(f"Model for {class_name} has been loaded")

    val_image_paths = glob.glob(os.path.join("datasets", "segmented", "Multiclass", "validvalid","images", "*"))
    val_mask_paths = glob.glob(os.path.join("datasets", "segmented", "Multiclass", "validvalid","labels", "*"))

    # for testing and debuging
    # val_image_paths= val_image_paths[:3]
    # val_mask_paths= val_mask_paths[:3]

    _, val_transform = get_transforms(input_size)
    val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    models_list = [models_dict[key] for key in class_names]
    
    probs6, gts = cache_probs(models_list, val_loader, device)
    mb = MetricBundle(num_classes=7, device=device)
    results = []
    for order in tqdm(itertools.permutations(range(6))):  
        # order = [0,5,4,1,3,2]
        preds = ordered_overlay(probs6, order)
        res   = compute_metrics_torchmetrics(preds, gts, mb, label_map=label_map)
        results.append((res["mean_iou"], order, res))
        # break
    topk = 3
    results.sort(key=lambda x: x[0], reverse=True)    
    print("Best mIoU:", results[0][0])
    print("Best order", [label_map[i+1] for i in results[0][1]])
    print("Per Category IoU", results[0][2]["per_category_iou"])
    print("Per Category PA", results[0][2]["pixel_accuracy_per_class"])
    print("mPA", results[0][2]["mean_pixel_accuracy"])
    print("Done")
    results_dict = {
        "best_mIoU": results[0][0],
        "best_order": [label_map[i+1] for i in results[0][1]],
        "order_indices": [i+1 for i in results[0][1]],
        "per_category_iou": list(map(float, results[0][2]["per_category_iou"])),
        "per_category_pa": list(map(float, results[0][2]["pixel_accuracy_per_class"])),
        "mean_pixel_accuracy": float(results[0][2]["mean_pixel_accuracy"])
    }

    # Create output directory
    output_path = os.path.join("results", "phase-2")
    os.makedirs(output_path, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(output_path, "phase2_results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {json_path}")

    preds = ordered_overlay(probs6, [i for i in results[0][1]])
    preds = preds.detach().cpu().numpy().astype(np.uint8)

    label_map =["Background","Mud","Sand","Shellhash coverage","Dog Cockle Bed","PatchesWorms","Bryozoans"]
    color_map = {
        'Mud':  (49, 46, 166),
        'Sand': (255, 117, 10),
        'Shellhash coverage': (0, 117, 239),      
        'Dog Cockle Bed': (67, 225, 65),      
        'PatchesWorms': (80, 212, 255),      
        'Bryozoans': (164, 1, 236),     
        }
    show_scale_percentage =50
    for index, img_path in enumerate(val_image_paths):
        frame = cv2.imread(img_path)
        w = int(frame.shape[1] * show_scale_percentage / 100)
        h = int(frame.shape[0] * show_scale_percentage / 100)
        frame = cv2.resize(frame, (w, h), interpolation = cv2.INTER_AREA)
        overlayed = np.zeros_like(frame)
        height,width, = frame.shape[:2]
        image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

        output_mask = cv2.imread(val_mask_paths[index],0)//30
        masks_dict = dict()
        for j, class_name in enumerate(label_map):
            msk = (output_mask==j).astype(np.uint8) * 255
            msk = cv2.resize(msk, (width, height))
            masks_dict[class_name] = msk.copy()
        overlayed = frame.copy()
        for class_name, mask in masks_dict.items():
            if class_name == 'Background':
                continue 
            color = color_map[class_name]
            colored_mask = np.zeros_like(overlayed)
            for c in range(3):
                colored_mask[:, :, c] = (mask > 0) * color[c]
            overlayed = cv2.addWeighted(overlayed, 1.0, colored_mask, 0.4, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlayed, contours, -1, (0,0,0), thickness=2)
        out = overlayed
        w = int(out.shape[1] * show_scale_percentage / 100)
        h = int(out.shape[0] * show_scale_percentage / 100)
        out = cv2.resize(out, (w, h), interpolation = cv2.INTER_AREA)

        output_path = os.path.join("results", "phase-2", "VAL", "GT")
        os.makedirs(output_path,exist_ok=True)
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)),out)

        output_mask = preds[index]
        masks_dict = dict()
        for j, class_name in enumerate(label_map):
            msk = (output_mask==j).astype(np.uint8) * 255
            msk = cv2.resize(msk, (width, height))
            masks_dict[class_name] = msk.copy()
        overlayed = frame.copy()
        for class_name, mask in masks_dict.items():
            if class_name == 'Background':
                continue 
            color = color_map[class_name]
            colored_mask = np.zeros_like(overlayed)
            for c in range(3):
                colored_mask[:, :, c] = (mask > 0) * color[c]
            overlayed = cv2.addWeighted(overlayed, 1.0, colored_mask, 0.4, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlayed, contours, -1, (0,0,0), thickness=2)
        out = overlayed
        w = int(out.shape[1] * show_scale_percentage / 100)
        h = int(out.shape[0] * show_scale_percentage / 100)
        out = cv2.resize(out, (w, h), interpolation = cv2.INTER_AREA)

        output_path = os.path.join("results", "phase-2", "VAL", "Preds")
        os.makedirs(output_path,exist_ok=True)
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)),out)


        


