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
from torchmetrics.classification import JaccardIndex
from collections import Counter
import argparse
from PIL import Image
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
import random
from featup.upsamplers import JBUStack
import evaluate
import copy
import json
from torchmetrics.classification import MulticlassJaccardIndex
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy
from sklearn.metrics import confusion_matrix
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor

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
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask[mask > 200] = 255
        mask[mask!=255] = mask[mask!=255]//MASK_SCALE
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


if __name__ == "__main__":
    seg_model_names = ["SegformerB0", "SegformerB2","SegformerB4","DeepLab","DeepLabPlus", "Unet", "Unetplusplus", "DPT", "UPerNet", "MAnet"]
    seg_model_names = ["Unetplusplus", "DPT", "UPerNet", "MAnet"]
    seg_model_names = ["Mask2Former"]
    dataset_version = "V3"
    for seg_model_name in seg_model_names:
        model_name = seg_model_name+"Dataset"+dataset_version+".pth"
        if "Segformer" in seg_model_name:
            input_size = 518
            if seg_model_name=="SegformerB4":
                model_name = "SegformerDataset"+dataset_version+".pth"
        elif seg_model_name == "Mask2Former":
            model_name = "Mask2FormerDataset"+dataset_version+".ckpt"
            input_size = 512

        elif seg_model_name=="DeepLab" or seg_model_name=="DeepLabPlus":
            input_size = 528
        else:
            input_size = 512

        label_map =["Background","Mud","Sand","Shellhash coverage","Dog Cockle Bed","PatchesWorms","Bryozoans"]
        NUM_CLASSES = len(label_map)
        id2label = {i: label_map[i] for i in range(NUM_CLASSES)}
        label2id = {v: k for k, v in id2label.items()}
        batch_size = 4
        model_path = os.path.join("trained_models", "multiclass", model_name)
        val_image_paths = glob.glob(os.path.join("datasets", "segmented", "Multiclass", "validvalid","images", "*"))
        val_mask_paths = glob.glob(os.path.join("datasets", "segmented", "Multiclass", "validvalid","labels", "*"))

        # val_image_paths.sort()
        # val_mask_paths.sort()
        # seed = 42
        # combined = list(zip(val_image_paths, val_mask_paths))
        # random.seed(seed)
        # random.shuffle(combined)
        # midpoint = len(combined) // 2
        # set_A = combined[:midpoint]
        # set_B = combined[midpoint:]
        # val_image_paths, val_mask_paths = zip(*set_B)

        if "Segformer" in seg_model_name:
            if seg_model_name == "SegformerB0":
                SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
            elif seg_model_name == "SegformerB2":
                SEGFORMER_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
            elif seg_model_name == "SegformerB4":
                SEGFORMER_MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
            model = SegformerForSemanticSegmentation.from_pretrained(
                SEGFORMER_MODEL_NAME,
                num_labels=NUM_CLASSES,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                ) 
        elif seg_model_name == "Mask2Former":
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-base-ade-semantic",
                id2label=id2label,
                label2id=label2id,
                num_labels=NUM_CLASSES,
                ignore_mismatched_sizes=True)
            Mask2FormerProcessor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",ignore_index=255,reduce_labels=False)
        elif seg_model_name=="DeepLab":
            backbone_name = "resnet101"
            model = smp.DeepLabV3Plus(encoder_name=backbone_name,
                                encoder_weights = "imagenet",
                                classes = NUM_CLASSES)    
        elif seg_model_name=="DeepLabPlus":
            backbone_name = "resnet101"
            model = smp.DeepLabV3Plus(encoder_name=backbone_name,
                                encoder_weights = "imagenet",
                                classes = NUM_CLASSES)
        elif seg_model_name=="Unetplusplus":
            backbone_name = "resnet101"
            model = smp.UnetPlusPlus(encoder_name=backbone_name,
                                encoder_weights = "imagenet",
                                classes = NUM_CLASSES)
        elif seg_model_name=="Unet":
            backbone_name = "resnet101"
            model = smp.Unet(encoder_name=backbone_name,
                                encoder_weights = "imagenet",
                                classes = NUM_CLASSES)
            
        elif seg_model_name=="DPT":
            backbone_name = "tu-resnet101"
            model = smp.DPT(encoder_name=backbone_name,
                                encoder_weights = "imagenet",
                                classes = NUM_CLASSES)
            model_name = seg_model_name+"Dataset"+dataset_version
        elif seg_model_name=="UPerNet":
            backbone_name = "resnet101"
            model = smp.UPerNet(encoder_name=backbone_name,
                                encoder_weights = "imagenet",
                                classes = NUM_CLASSES)
        elif seg_model_name=="MAnet":
            backbone_name = "resnet101"
            model = smp.MAnet(encoder_name=backbone_name,
                                encoder_weights = "imagenet",
                                classes = NUM_CLASSES)
        else:
            raise ValueError("Model name is not valid ...")
        _, val_transform = get_transforms(input_size)
        if seg_model_name == "Mask2Former":
            def collate_fn(batch):
                inputs = list(zip(*batch))
                images=inputs[0]
                segmentation_maps=inputs[1]
                batch = Mask2FormerProcessor(
                    images,
                    segmentation_maps=segmentation_maps,
                    size=(512,512),
                    return_tensors="pt",
                )
                batch["original_images"] = images
                batch["original_segmentation_maps"] = segmentation_maps
                return batch
            val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=None)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        else:
            val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        truth_masks = []
        for path in val_mask_paths:
            truth_masks.append(cv2.resize(cv2.imread(path,0)//30, (input_size,input_size)))
        if seg_model_name == "Mask2Former":
            ckpt = torch.load(model_path, map_location="cuda")
            raw_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            cleaned_sd = {}
            for k, v in raw_sd.items():
                k2 = k[6:] if k.startswith("model.") else k
                if k2.startswith("criterion.") or k2.startswith("loss."):
                    continue
                cleaned_sd[k2] = v
            incompat = model.load_state_dict(cleaned_sd, strict=False)
            print(f"[Mask2Former] loaded with missing={len(incompat.missing_keys)} "
                f"unexpected={len(incompat.unexpected_keys)}")
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        label_map = ["Background", "Mud", "Sand", "Shellhash coverage", "Dog Cockle Bed", "PatchesWorms", "Bryozoans"]
        metric = evaluate.load("mean_iou")
        labels_all = []
        preds_all = []

        if seg_model_name == "Mask2Former":
            
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    pixel_values = batch["pixel_values"].to(device)       
                    B, _, Hf, Wf = pixel_values.shape
                    outputs = model(pixel_values=pixel_values)
                    pred_maps = Mask2FormerProcessor.post_process_semantic_segmentation(
                        outputs=outputs,
                        target_sizes=[(Hf, Wf)] * B)  
                    for pred, gt_np in zip(pred_maps, batch["original_segmentation_maps"]):
                        gt_resized = cv2.resize(gt_np.astype(np.uint8),(Wf, Hf),interpolation=cv2.INTER_NEAREST).astype(np.int64)   
                        preds_all.append(pred.cpu().numpy())
                        labels_all.append(gt_resized)
        else:
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc="Evaluating"):
                    images = images.to(device)
                    masks = masks.to(device)
                    if "Segformer" in seg_model_name:
                        outputs = model(images).logits
                        outputs = F.interpolate(outputs, size=(input_size, input_size), mode="bilinear", align_corners=False)
                    else:
                        outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    preds_all.extend(preds.cpu().numpy())
                    labels_all.extend(masks.cpu().numpy())

        preds_np = np.array(preds_all)
        labels_np = np.array(labels_all)

        preds_flat = torch.tensor(preds_np).reshape(-1)
        labels_flat = torch.tensor(labels_np).reshape(-1)
        mask = labels_flat != 255
        preds_flat = preds_flat[mask]
        labels_flat = labels_flat[mask]

        jaccard = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average=None).to(device)
        dice = MulticlassF1Score(num_classes=NUM_CLASSES, average=None).to(device)
        dice_mean = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)
        pixel_acc = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro").to(device)
        mean_acc = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro").to(device)

        # Weighted mIoU
        jaccard_w = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="weighted").to(device)
        miou_weighted = jaccard_w(preds_flat.to(device), labels_flat.to(device)).item()

        # Weighted pixel accuracy (PA)
        acc_w = MulticlassAccuracy(num_classes=NUM_CLASSES, average="weighted").to(device)
        pa_weighted = acc_w(preds_flat.to(device), labels_flat.to(device)).item()

        per_class_iou = jaccard(preds_flat.to(device), labels_flat.to(device)).cpu().tolist()
        per_class_dice = dice(preds_flat.to(device), labels_flat.to(device)).cpu().tolist()
        mean_dice = dice_mean(preds_flat.to(device), labels_flat.to(device)).item()
        pix_acc = pixel_acc(preds_flat.to(device), labels_flat.to(device)).item()
        mean_class_acc = mean_acc(preds_flat.to(device), labels_flat.to(device)).item()

        cm = confusion_matrix(labels_flat.cpu(), preds_flat.cpu(), labels=list(range(NUM_CLASSES)))
        correct_per_class = cm.diagonal()
        total_per_class = cm.sum(axis=1)
        pa_per_class = (correct_per_class / total_per_class).tolist()
        mean_pa = float(np.mean([pa for pa in pa_per_class if not np.isnan(pa)]))

        results_dict = {
            "label_map": label_map,
            "mean_iou": float(np.mean(per_class_iou)),
            "per_category_iou": per_class_iou,
            "per_category_dice": per_class_dice,
            "mean_dice": mean_dice,
            "pixel_accuracy": pix_acc,
            "mean_class_accuracy": mean_class_acc,
            "pixel_accuracy_per_class": pa_per_class,
            "mean_pixel_accuracy": mean_pa,
            "mean_iou_weighted": miou_weighted,
            "pixel_accuracy_weighted": pa_weighted,
        }

        # Create output directory
        output_path = os.path.join("results", "phase-3", model_name)
        os.makedirs(output_path, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(output_path, model_name.replace(".pth", "_metrics.json"))
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results saved to {json_path}")


        preds = preds_np.copy().astype(np.uint8)
        label_map =["Background","Mud","Sand","Shellhash coverage","Dog Cockle Bed","PatchesWorms","Bryozoans"]
        def darken_color(color, factor=1):
            return tuple(int(c * factor) for c in color)

        color_map = {
            'Mud':  darken_color((49, 46, 166)),
            'Sand': darken_color((255, 117, 10)),
            'Shellhash coverage': darken_color((0, 117, 239)),      
            'Dog Cockle Bed': darken_color((67, 225, 65)),      
            'PatchesWorms': darken_color((20, 255, 255)),      
            'Bryozoans': darken_color((164, 1, 236)),     
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
            # legend_start_x, legend_start_y = 10, 10  # Top-left corner
            # box_size = 20
            # spacing = 10
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 1
            # font_thickness = 1

            # for i, (class_name, color) in enumerate(color_map.items()):
            #     y = legend_start_y + i * (box_size + spacing)
            #     cv2.rectangle(overlayed, (legend_start_x, y),
            #                 (legend_start_x + box_size, y + box_size),
            #                 color, thickness=-1)
            #     cv2.putText(overlayed, class_name,
            #                 (legend_start_x + box_size + 10, y + box_size - 5),
            #                 font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            out = overlayed
            w = int(out.shape[1] * show_scale_percentage / 100)
            h = int(out.shape[0] * show_scale_percentage / 100)
            out = cv2.resize(out, (w, h), interpolation = cv2.INTER_AREA)

            output_path_ = os.path.join(output_path, "GT")
            os.makedirs(output_path_,exist_ok=True)
            cv2.imwrite(os.path.join(output_path_, os.path.basename(img_path)),out)



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
            # legend_start_x, legend_start_y = 10, 10  # Top-left corner
            # box_size = 20
            # spacing = 10
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 1
            # font_thickness = 1

            # for i, (class_name, color) in enumerate(color_map.items()):
            #     y = legend_start_y + i * (box_size + spacing)
            #     cv2.rectangle(overlayed, (legend_start_x, y),
            #                 (legend_start_x + box_size, y + box_size),
            #                 color, thickness=-1)
            #     cv2.putText(overlayed, class_name,
            #                 (legend_start_x + box_size + 10, y + box_size - 5),
            #                 font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            out = overlayed
            w = int(out.shape[1] * show_scale_percentage / 100)
            h = int(out.shape[0] * show_scale_percentage / 100)
            out = cv2.resize(out, (w, h), interpolation = cv2.INTER_AREA)

            output_path_ = os.path.join(output_path, "Preds")
            os.makedirs(output_path_,exist_ok=True)
            cv2.imwrite(os.path.join(output_path_, os.path.basename(img_path)),out)
